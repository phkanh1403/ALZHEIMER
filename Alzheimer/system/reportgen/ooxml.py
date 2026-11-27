"""Put pandoc's Word XML in schema order.

pandoc 3.1 writes some child elements out of the order the OOXML schema requires (style options, paragraph and
table properties, settings, a few math properties) and short list ids. Word usually tolerates this, but strict
validators do not; this pass reorders the children of those elements and pads the ids, without changing content.
"""
import shutil
import tempfile
import zipfile
from pathlib import Path

from lxml import etree

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'

# child order of each element (local names), from the ECMA-376 schema
ORDER = {
    (W, 'style'): ['name', 'aliases', 'basedOn', 'next', 'link', 'autoRedefine', 'hidden', 'uiPriority',
                   'semiHidden', 'unhideWhenUsed', 'qFormat', 'locked', 'personal', 'personalCompose',
                   'personalReply', 'rsid', 'pPr', 'rPr', 'tblPr', 'trPr', 'tcPr', 'tblStylePr'],
    (W, 'pPr'): ['pStyle', 'keepNext', 'keepLines', 'pageBreakBefore', 'framePr', 'widowControl', 'numPr',
                 'suppressLineNumbers', 'pBdr', 'shd', 'tabs', 'suppressAutoHyphens', 'kinsoku', 'wordWrap',
                 'overflowPunct', 'topLinePunct', 'autoSpaceDE', 'autoSpaceDN', 'bidi', 'adjustRightInd',
                 'snapToGrid', 'spacing', 'ind', 'contextualSpacing', 'mirrorIndents', 'suppressOverlap', 'jc',
                 'textDirection', 'textAlignment', 'textboxTightWrap', 'outlineLvl', 'divId', 'cnfStyle', 'rPr',
                 'sectPr', 'pPrChange'],
    (W, 'tcPr'): ['cnfStyle', 'tcW', 'gridSpan', 'hMerge', 'vMerge', 'tcBorders', 'shd', 'noWrap', 'tcMar',
                  'textDirection', 'tcFitText', 'vAlign', 'hideMark'],
    (W, 'tblPr'): ['tblStyle', 'tblpPr', 'tblOverlap', 'bidiVisual', 'tblStyleRowBandSize', 'tblStyleColBandSize',
                   'tblW', 'jc', 'tblCellSpacing', 'tblInd', 'tblBorders', 'shd', 'tblLayout', 'tblCellMar',
                   'tblLook', 'tblCaption', 'tblDescription'],
    (W, 'rPr'): ['ins', 'del', 'moveFrom', 'moveTo', 'rStyle', 'rFonts', 'b', 'bCs', 'i', 'iCs', 'caps',
                 'smallCaps', 'strike', 'dstrike', 'outline', 'shadow', 'emboss', 'imprint', 'noProof', 'snapToGrid',
                 'vanish', 'webHidden', 'color', 'spacing', 'w', 'kern', 'position', 'sz', 'szCs', 'highlight', 'u',
                 'effect', 'bdr', 'shd', 'fitText', 'vertAlign', 'rtl', 'cs', 'em', 'lang', 'eastAsianLayout',
                 'specVanish', 'oMath'],
    (W, 'settings'): [
        'writeProtection', 'view', 'zoom', 'removePersonalInformation', 'removeDateAndTime',
        'doNotDisplayPageBoundaries', 'displayBackgroundShape', 'printPostScriptOverText',
        'printFractionalCharacterWidth', 'printFormsData', 'embedTrueTypeFonts', 'embedSystemFonts',
        'saveSubsetFonts', 'saveFormsData', 'mirrorMargins', 'alignBordersAndEdges', 'bordersDoNotSurroundHeader',
        'bordersDoNotSurroundFooter', 'gutterAtTop', 'hideSpellingErrors', 'hideGrammaticalErrors',
        'activeWritingStyle', 'proofState', 'formsDesign', 'attachedTemplate', 'linkStyles', 'stylePaneFormatFilter',
        'stylePaneSortMethod', 'documentType', 'mailMerge', 'revisionView', 'trackRevisions', 'doNotTrackMoves',
        'doNotTrackFormatting', 'documentProtection', 'autoFormatOverride', 'styleLockTheme', 'styleLockQFSet',
        'defaultTabStop', 'autoHyphenation', 'consecutiveHyphenLimit', 'hyphenationZone', 'doNotHyphenateCaps',
        'showEnvelope', 'summaryLength', 'clickAndTypeStyle', 'defaultTableStyle', 'evenAndOddHeaders',
        'bookFoldRevPrinting', 'bookFoldPrinting', 'bookFoldPrintingSheets', 'drawingGridHorizontalSpacing',
        'drawingGridVerticalSpacing', 'displayHorizontalDrawingGridEvery', 'displayVerticalDrawingGridEvery',
        'doNotUseMarginsForDrawingGridOrigin', 'drawingGridHorizontalOrigin', 'drawingGridVerticalOrigin',
        'doNotShadeFormData', 'noPunctuationKerning', 'characterSpacingControl', 'printTwoOnOne',
        'strictFirstAndLastChars', 'noLineBreaksAfter', 'noLineBreaksBefore', 'savePreviewPicture',
        'doNotValidateAgainstSchema', 'saveInvalidXml', 'ignoreMixedContent', 'alwaysShowPlaceholderText',
        'doNotDemarcateInvalidXml', 'saveXmlDataOnly', 'useXSLTWhenSaving', 'saveThroughXslt', 'showXMLTags',
        'alwaysMergeEmptyNamespace', 'updateFields', 'hdrShapeDefaults', 'footnotePr', 'endnotePr', 'compat',
        'docVars', 'rsids', 'mathPr', 'attachedSchema', 'themeFontLang', 'clrSchemeMapping',
        'doNotIncludeSubdocsInStats', 'doNotAutoCompressPictures', 'forceUpgrade', 'captions',
        'readModeInkLockDown', 'smartTagType', 'schemaLibrary', 'shapeDefaults', 'doNotEmbedSmartTags',
        'decimalSymbol', 'listSeparator'],
    (M, 'dPr'): ['begChr', 'sepChr', 'endChr', 'grow', 'shp', 'ctrlPr'],
    (M, 'rPr'): ['lit', 'nor', 'scr', 'sty', 'brk', 'aln'],
    (M, 'mcPr'): ['count', 'mcJc'],
}
PARTS = ['word/document.xml', 'word/styles.xml', 'word/settings.xml', 'word/numbering.xml']


def _reorder(root):
    for (ns, tag), order in ORDER.items():
        rank = {name: i for i, name in enumerate(order)}
        for parent in root.iter(f'{{{ns}}}{tag}'):
            children = list(parent)
            keyed = sorted(enumerate(children),
                           key=lambda ic: (rank.get(etree.QName(ic[1]).localname, len(order)), ic[0]))
            if [c for _, c in keyed] != children:
                for c in children:
                    parent.remove(c)
                for _, c in keyed:
                    parent.append(c)


def _clean(root):
    """Drop stray text between property elements and math style options that 'normal text' (m:nor) excludes."""
    for rpr in root.iter(f'{{{W}}}rPr'):
        for child in rpr:
            if child.tail and child.tail.strip():
                child.tail = None
    for rpr in root.iter(f'{{{M}}}rPr'):
        if rpr.find(f'{{{M}}}nor') is not None:
            for tag in ('scr', 'sty'):
                for el in rpr.findall(f'{{{M}}}{tag}'):
                    rpr.remove(el)


def _pad_list_ids(root):
    """w:nsid and w:tmpl need eight hex digits."""
    for tag in ('nsid', 'tmpl'):
        for el in root.iter(f'{{{W}}}{tag}'):
            value = el.get(f'{{{W}}}val')
            if value and len(value) < 8:
                el.set(f'{{{W}}}val', value.zfill(8).upper())


def fix(path):
    """Rewrite the DOCX at path in place."""
    path = Path(path)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / path.name
        with zipfile.ZipFile(path) as src, zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as dst:
            for item in src.infolist():
                if item.filename.endswith('/'):
                    continue                     # folder entries confuse some Word co-editing services
                data = src.read(item.filename)
                if item.filename in PARTS:
                    root = etree.fromstring(data)
                    _clean(root)
                    _reorder(root)
                    _pad_list_ids(root)
                    data = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
                dst.writestr(item, data)
        shutil.move(out, path)
