"""Download public, accepted-paper full texts used by the research memo."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parent
SOURCES = {
    'Receler_ECCV2024': 'https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05685.pdf',
    'SPM_CVPR2024': 'https://openaccess.thecvf.com/content/CVPR2024/papers/Lyu_One-dimensional_Adapter_to_Rule_Them_All_Concepts_Diffusion_Models_and_CVPR_2024_paper.pdf',
    'AdaCD_ACL2026': 'https://aclanthology.org/2026.acl-long.1823.pdf',
    'SteerMoE_ICLR2026': 'https://proceedings.iclr.cc/paper_files/paper/2026/file/0d61c5f5ef91e7e8a091b7b8f72b853c-Paper-Conference.pdf',
    'SafeGuide_ICLR2026_Workshop': 'https://openreview.net/pdf?id=kjUisljQle',
    # Accepted workshop paper; public author version because the venue PDF returns 403.
    'DSA_ICLR2026_Workshop': 'https://arxiv.org/pdf/2506.00166',
    'SafeDecoding_ACL2024': 'https://aclanthology.org/2024.acl-long.303.pdf',
    'SafeInfer_AAAI2025': 'https://ojs.aaai.org/index.php/AAAI/article/download/34927/37082',
    'HiddenGuard_ACL2026': 'https://aclanthology.org/2026.acl-long.1482.pdf',
    'SafeSteer_FindingsACL2026': 'https://aclanthology.org/2026.findings-acl.916.pdf',
    'ABack_FindingsACL2026': 'https://aclanthology.org/2026.findings-acl.1857.pdf',
    'SafeCompass_ICML2026': 'https://raw.githubusercontent.com/mlresearch/v306/main/assets/zhang26hc/zhang26hc.pdf',
    'FKSteering_ICML2025': 'https://raw.githubusercontent.com/mlresearch/v267/main/assets/singhal25b/singhal25b.pdf',
    'ACoFi_L4DC2026': 'https://raw.githubusercontent.com/mlresearch/v331/main/assets/huriot26a/huriot26a.pdf',
    'SteeringSafety_ICML2026': 'https://raw.githubusercontent.com/mlresearch/v306/main/assets/siu26a/siu26a.pdf',
    'SteeringSafely_EACL2026': 'https://aclanthology.org/2026.eacl-long.268.pdf',
    'SAeUron_ICML2025': 'https://raw.githubusercontent.com/mlresearch/v267/main/assets/cywinski25a/cywinski25a.pdf',
    'GIE_ICLR2025': 'https://proceedings.iclr.cc/paper_files/paper/2025/file/c48dcd605cfb17d5b4f94ce106a915f7-Paper-Conference.pdf',
}


def fetch(item):
    key, url = item
    pdf = ROOT / 'sources' / 'pdfs' / f'{key}.pdf'
    txt = ROOT / 'sources' / 'text' / f'{key}.txt'
    pdf.parent.mkdir(parents=True, exist_ok=True)
    txt.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not pdf.exists():
            request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(request, timeout=45) as response:
                data = response.read()
            if not data.startswith(b'%PDF'):
                raise ValueError('Response is not a PDF')
            pdf.write_bytes(data)
        subprocess.run(['pdftotext', '-layout', str(pdf), str(txt)], check=True)
        return {'id': key, 'pdf_url': url, 'local_pdf': str(pdf.relative_to(ROOT)),
                'local_text': str(txt.relative_to(ROOT)), 'bytes': pdf.stat().st_size,
                'sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(), 'status': 'ok'}
    except Exception as error:
        return {'id': key, 'pdf_url': url, 'status': 'error', 'error': str(error)}


if __name__ == '__main__':
    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(fetch, item) for item in SOURCES.items()]):
            record = future.result()
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    (ROOT / 'sources' / 'downloads.json').write_text(
        json.dumps(sorted(records, key=lambda r: r['id']), indent=2, ensure_ascii=False) + '\n')
