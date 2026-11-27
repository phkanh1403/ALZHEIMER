"""Turn the IEEE LaTeX report into a single-column Word document with pandoc.

pandoc does not know IEEEtran, algorithmicx or cross-reference numbers, so the source is first rewritten into plain
LaTeX: numbers, labels and citations are resolved from the .aux file of the PDF build, so the Word file carries the
same section, figure, table, equation and reference numbers as the PDF.
"""
import re
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

from . import ooxml

ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV']
THEOREMS = {'definition': 'Định nghĩa', 'proposition': 'Mệnh đề', 'lemma': 'Bổ đề', 'remark': 'Nhận xét'}
FONT = 'Times New Roman'


# --------------------------------------------------------------------------- helpers
def braced(text, start):
    """(content, end) of the brace group opening at text[start] == '{'."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
    raise ValueError(f'unbalanced braces at {start}')


def replace_command(text, name, render):
    """Replace every \\name{arg} by render(arg)."""
    out, pos = [], 0
    pattern = re.compile(r'\\' + name + r'\s*\{')
    while (m := pattern.search(text, pos)):
        arg, end = braced(text, m.end() - 1)
        out += [text[pos:m.start()], render(arg)]
        pos = end
    return ''.join([*out, text[pos:]])


def read_aux(path):
    """Label -> number and citation key -> number from a LaTeX .aux file."""
    labels, cites = {}, {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.startswith('\\newlabel{'):
            key, end = braced(line, len('\\newlabel'))
            value, _ = braced(line, end)
            number, _ = braced(value, 0)
            labels[key] = number
        elif line.startswith('\\bibcite{'):
            key, end = braced(line, len('\\bibcite'))
            cites[key] = braced(line, end)[0]
    return labels, cites


# --------------------------------------------------------------------------- rewriting steps
def inline_inputs(text, base):
    return replace_command(text, 'input', lambda arg: (base / f'{arg}.tex').read_text(encoding='utf-8')
                           if not arg.endswith('numbers') else '')


def expand_numbers(text, numbers):
    def value(key):
        if key not in numbers:
            raise KeyError(f'no number for \\R{{{key}}}')
        return numbers[key]
    return replace_command(text, 'R', value)


def resolve_refs(text, labels, cites):
    text = replace_command(text, 'eqref', lambda k: f'({labels[k]})')
    text = replace_command(text, 'ref', lambda k: labels[k])
    return replace_command(text, 'cite', lambda keys: ', '.join(f'[{cites[k.strip()]}]' for k in keys.split(',')))


def number_sections(text):
    """Prefix headings with IEEE numbers: sections I, II, ...; subsections A, B, ...; appendices A, B, ..."""
    out, pos, sec, sub, appendix = [], 0, 0, 0, False
    for m in re.finditer(r'\\appendices|\\(section|subsection)\*?\s*\{', text):
        out.append(text[pos:m.start()])
        if m.group(0) == '\\appendices':
            appendix, sec, pos = True, 0, m.end()
            continue
        title, end = braced(text, m.end() - 1)
        if m.group(1) == 'section':
            sec, sub = sec + 1, 0
            label = f'Phụ lục {chr(64 + sec)}: ' if appendix else f'{ROMAN[sec - 1]}. '
            out.append(f'\\section*{{{label}{title}}}')
        else:
            sub += 1
            out.append(f'\\subsection*{{{chr(64 + sub)}. {title}}}')
        pos = end
    return ''.join([*out, text[pos:]])


def theorems(text):
    for env, name in THEOREMS.items():
        count = 0

        def head(m, name=name):
            nonlocal count
            count += 1
            note = f' ({m.group(1)})' if m.group(1) else ''
            return f'\\par\\emph{{{name} {count}{note}:}} '
        text = re.sub(r'\\begin\{' + env + r'\}(?:\[([^\]]*)\])?(?:\\label\{[^}]*\})?', head, text)
        text = text.replace(f'\\end{{{env}}}', '\\par')
    text = re.sub(r'\\begin\{IEEEproof\}', r'\\par\\emph{Chứng minh:} ', text)
    return text.replace('\\end{IEEEproof}', ' ∎\\par')


def equations(text, labels):
    """Equation labels become a visible '(n)' at the end of the equation: Word has no automatic equation numbers and
    pandoc drops \\tag."""
    def single(m):
        env, body = m.group(1), m.group(2)
        label = re.search(r'\\label\{(eq:[^}]*)\}', body)
        if not label:
            return m.group(0)
        body = body.replace(label.group(0), '').rstrip()
        return f'\\begin{{{env}}}{body}\\qquad({labels[label.group(1)]})\n\\end{{{env}}}'
    text = re.sub(r'\\begin\{(equation|multline)\}(.*?)\\end\{\1\}', single, text, flags=re.S)
    # multi-line environments: labels sit at the end of their line
    return re.sub(r'\\label\{(eq:[^}]*)\}', lambda m: f'\\qquad({labels[m.group(1)]})', text)


def floats(text, labels):
    """Numbered captions for figures and tables; PNG images instead of PDF."""
    def caption(body, prefix):
        label = re.search(r'\\label\{([^}]*)\}', body)
        number = labels[label.group(1)] if label else ''
        return replace_command(body, 'caption', lambda c: f'\\caption{{{prefix} {number}. {c}}}')

    def figure(m):
        body = m.group(2).replace('.pdf}', '.png}')
        return f'\\begin{{figure}}{caption(body, "Hình")}\\end{{figure}}'

    def table(m):
        return f'\\begin{{table}}{caption(m.group(2), "Bảng")}\\end{{table}}'
    text = re.sub(r'\\begin\{figure(\*?)\}(?:\[[^\]]*\])?(.*?)\\end\{figure\*?\}', figure, text, flags=re.S)
    return re.sub(r'\\begin\{table(\*?)\}(?:\[[^\]]*\])?(.*?)\\end\{table\*?\}', table, text, flags=re.S)


def math_for_pandoc(text):
    """Drop manual delimiter sizes (\\Bigl, \\bigr, ...), which pandoc's math reader rejects; Word sizes them."""
    return re.sub(r'\\[Bb]ig{1,2}[lr]?(?![a-zA-Z])', '', text)


def _simple_columns(spec):
    """'@{}lL{0.6\\columnwidth}r@{}' -> 'llr': pandoc only needs the alignment of each column."""
    out, i = [], 0
    while i < len(spec):
        c = spec[i]
        if c in '@>':
            _, i = braced(spec, i + 1)
            continue
        if c in 'pLmb':
            _, i = braced(spec, i + 1)
            out.append('l')
            continue
        if c in 'lrc':
            out.append(c)
        i += 1
    return ''.join(out)


def tables_for_pandoc(text):
    out, pos = [], 0
    for m in re.finditer(r'\\begin\{tabular\}\{', text):
        if m.start() < pos:
            continue
        spec, end = braced(text, m.end() - 1)
        out += [text[pos:m.start()], f'\\begin{{tabular}}{{{_simple_columns(spec)}}}']
        pos = end
    text = ''.join([*out, text[pos:]])
    text = re.sub(r'\\multirow\{[^}]*\}\{[^}]*\}\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\cmidrule(\([^)]*\))?\{[^}]*\}', '', text)
    return re.sub(r'\\setlength\{[^}]*\}\{[^}]*\}', '', text)


def algorithms(text, labels):
    """algorithmicx blocks become a bold title and indented numbered steps."""
    def convert(m):
        body = m.group(1)
        label = re.search(r'\\label\{([^}]*)\}', body)
        cap, _ = braced(body, body.index('\\caption') + len('\\caption'))
        lines, depth, out = body.split('\n'), 0, []
        for raw in lines:
            line = replace_command(raw.strip(), 'Comment', lambda c: f' ▷ {c}')
            if line.startswith('\\Require'):
                out.append(f'\\textbf{{Đầu vào:}} {line[len(chr(92) + "Require"):].strip()}\\par')
            elif line.startswith('\\Ensure'):
                out.append(f'\\textbf{{Đầu ra:}} {line[len(chr(92) + "Ensure"):].strip()}\\par')
            elif line.startswith('\\For'):
                cond, rest = braced(line, line.index('{'))
                out.append(f'\\item {"~" * 4 * depth}\\textbf{{for}} {cond} \\textbf{{do}}{line[rest:]}')
                depth += 1
            elif line.startswith('\\EndFor'):
                depth -= 1
            elif line.startswith('\\State'):
                out.append(f'\\item {"~" * 4 * depth}{line[len(chr(92) + "State"):].strip()}')
        head = [i for i in out if not i.startswith('\\item')]
        steps = [i for i in out if i.startswith('\\item')]
        foot = [h for h in head if 'Đầu ra' in h]
        head = [h for h in head if 'Đầu ra' not in h]
        return (f'\\par\\textbf{{Thuật toán {labels[label.group(1)]}. {cap}}}\\par' + '\n'.join(head)
                + '\n\\begin{enumerate}\n' + '\n'.join(steps) + '\n\\end{enumerate}\n' + '\n'.join(foot))
    return re.sub(r'\\begin\{algorithm\}(?:\[[^\]]*\])?(.*?)\\end\{algorithm\}', convert, text, flags=re.S)


def front_matter(text):
    """Plain title block: title, author, abstract, keywords and the first-page note."""
    thanks = ''
    if '\\thanks' in text:
        start = text.index('\\thanks')
        thanks, end = braced(text, start + len('\\thanks'))
        text = text[:start] + text[end:]
    text = text.replace('\\begin{IEEEkeywords}', '\\par\\textbf{Từ khóa—}').replace('\\end{IEEEkeywords}', '\\par')
    text = text.replace('\\end{abstract}', f'\\end{{abstract}}\n\\emph{{{thanks}}}\\par\n', 1)
    text = re.sub(r'\\markboth\{[^}]*\}%?\s*\{[^}]*\}', '', text)
    return re.sub(r'%\n', '\n', text)


def bibliography(text):
    """thebibliography -> a numbered reference section (pandoc ignores \\bibitem numbers)."""
    m = re.search(r'\\begin\{thebibliography\}\{[^}]*\}(.*?)\\end\{thebibliography\}', text, flags=re.S)
    if not m:
        return text
    items = re.split(r'\\bibitem\{[^}]*\}', m.group(1))[1:]
    refs = '\n\n'.join(f'[{i}] {item.strip()}' for i, item in enumerate(items, 1))
    return text[:m.start()] + f'\\section*{{Tài liệu tham khảo}}\n{refs}\n' + text[m.end():]


def preamble(text):
    """Keep only what pandoc understands from the preamble (macro definitions and title data)."""
    head, body = text.split('\\begin{document}', 1)
    keep = [line for line in head.splitlines() if line.startswith(('\\newcommand', '\\title', '\\author'))]
    return '\\documentclass{article}\n' + '\n'.join(keep) + '\n\\begin{document}' + body


def to_pandoc_latex(source, numbers, aux, base):
    labels, cites = read_aux(aux)
    text = Path(source).read_text(encoding='utf-8')
    text = inline_inputs(text, base)
    text = expand_numbers(text, numbers)
    text = front_matter(text)
    text = bibliography(text)
    text = algorithms(text, labels)
    text = floats(text, labels)
    text = equations(text, labels)
    text = theorems(text)
    text = resolve_refs(text, labels, cites)
    text = number_sections(text)
    text = tables_for_pandoc(text)
    text = math_for_pandoc(text)
    for drop in ('\\footnotesize', '\\centering', '\\maketitle', '\\small'):
        text = text.replace(drop, '')
    return preamble(text)


# --------------------------------------------------------------------------- Word styling
SIZES = {'Normal': 11, 'Body Text': 11, 'First Paragraph': 11, 'Compact': 11, 'Title': 18, 'Heading 1': 13,
         'Heading 2': 11.5, 'Abstract': 10, 'Caption': 9.5, 'Image Caption': 9.5, 'Table Caption': 9.5, 'Author': 11,
         'Footnote Text': 9}
CENTRED = ('Image Caption', 'Table Caption', 'Captioned Figure', 'Title', 'Author', 'Abstract Title')
JUSTIFIED = ('Body Text', 'First Paragraph', 'Abstract')


def _set_font(style_):
    """Use FONT for every script, replacing theme fonts."""
    style_.font.name = FONT
    rpr = style_.element.get_or_add_rPr()
    fonts = rpr.find(qn('w:rFonts'))
    if fonts is None:
        fonts = OxmlElement('w:rFonts')
        rpr.append(fonts)
    for attr in ('w:ascii', 'w:hAnsi', 'w:eastAsia', 'w:cs'):
        fonts.set(qn(attr), FONT)
    for attr in ('w:asciiTheme', 'w:hAnsiTheme', 'w:eastAsiaTheme', 'w:cstheme'):
        fonts.attrib.pop(qn(attr), None)


def _style_text(doc):
    for s in doc.styles:
        if s.type != 1:          # paragraph and character styles only
            continue
        _set_font(s)
        if s.name in SIZES:
            s.font.size = Pt(SIZES[s.name])
        if s.name.startswith('Heading') or s.name == 'Title':
            s.font.color.rgb = RGBColor(0, 0, 0)
            s.font.bold = True
    for p in doc.paragraphs:
        if p.style.name == 'Abstract Title':        # pandoc writes the English heading
            for run in p.runs:
                run.text = ''
            p.runs[0].text = 'Tóm tắt'
        if p.style.name in CENTRED:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif p.style.name in JUSTIFIED:
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def _style_tables(doc):
    for table in doc.tables:
        borders = OxmlElement('w:tblBorders')
        for edge in ('top', 'bottom', 'insideH'):
            e = OxmlElement(f'w:{edge}')
            for key, value in (('w:val', 'single'), ('w:sz', '4'), ('w:color', '808080')):
                e.set(qn(key), value)
            borders.append(e)
        table._tbl.tblPr.append(borders)
        for cell in (c for row in table.rows for c in row.cells):
            for run in (r for p in cell.paragraphs for r in p.runs):
                run.font.size = Pt(9)


def style(path):
    """Times New Roman throughout, IEEE-like sizes, centred captions and bordered tables."""
    doc = Document(path)
    _style_text(doc)
    _style_tables(doc)
    doc.save(path)


def build(source, numbers, aux, out, resource_dir):
    """Write the DOCX next to the PDF; returns its path."""
    base = Path(source).parent
    tex = to_pandoc_latex(source, numbers, aux, base)
    tmp = Path(out).with_suffix('.pandoc.tex')
    tmp.write_text(tex, encoding='utf-8')
    subprocess.run(['pandoc', str(tmp), '-f', 'latex', '-t', 'docx', '-o', str(out),
                    f'--resource-path={resource_dir}'], check=True)
    tmp.unlink()
    style(out)
    ooxml.fix(out)
    return out
