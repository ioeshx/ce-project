"""Build provenance records for the three-section research memo from local evidence."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent
PAPERS = ROOT.parent / 'papers'
report = (ROOT / 'RESEARCH.zh-CN.md').read_text()
links = dict(re.findall(r'^\[([DX]\d+[A-Z]?)\]: (\S+)$', report, re.M))
legacy = {x['id']: x for x in json.loads((PAPERS / 'inference_no_projection/results.json').read_text())['records']}
inventory = json.loads((PAPERS / 'codes/inventory.json').read_text())
downloads = {x['id']: x for x in json.loads((ROOT / 'sources/downloads.json').read_text())}

records = []
old_map = {
    'D01': 'SLD', 'D02': 'DNG', 'D03': 'SafeDenoiser', 'D04': 'SGF',
    'D05': 'CASG+SLD', 'D06': 'STG', 'D08': 'ConceptCorrector',
    'D09': 'UVR', 'D13': 'SemanticSteering-MM', 'D18': 'SEGA',
    'D19': 'AMS', 'D20': 'IRECE',
}
for key, old_id in old_map.items():
    old = legacy[old_id]
    records.append({
        'id': key, 'title': old['title'], 'venue': old['venue'],
        'conference_date': old['date'], 'publication_status': old['status'],
        'primary_url': links[key], 'verification_depth': 'full text; formal acceptance provenance reused from prior audit',
        'evidence': old['evidence'], 'local_pdf': old['local_pdf'],
        'new_parameter_training': False,
        'scope_note': ('Related work; not an independent training-free erasure system' if key in ['D18','D19','D20'] else 'T2I core representative'),
    })

extra = [
    ('D07', 'Prototype-Guided Concept Erasure in Diffusion Models', 'CVPR 2026 main', '2026-06', True, None,
     'Section 3 and Algorithm 1: image prototypes and learned soft prompts; original backbone frozen.'),
    ('D10', 'Growth Inhibitors for Suppressing Inappropriate Image Concepts in Diffusion Models', 'ICLR 2025 main', '2025-04', True, 'GIE_ICLR2025',
     'Sections 4.1–4.3; MLP predicts suppression scale; multi-concept semantic deduplication.'),
    ('D11', 'Detect-and-Guide: Self-regulation of Diffusion Models for Safe Text-to-Image Generation via Guideline Token Optimization', 'CVPR 2025 main', '2025-06', True, None,
     'Sections 3.1–3.2: learned guideline tokens and layer weights; spatial adaptive guidance.'),
    ('D12', 'Concept Erasure via Attention Redirection', 'CVPR Findings 2026', '2026-06', True, None,
     'Sections 3.1–3.2: frozen FLUX, learned eraser key and scale, preservation and reconstruction losses.'),
    ('D14', 'SAeUron: Interpretable Concept Unlearning in Diffusion Models with Sparse Autoencoders', 'ICML 2025 main', '2025-07', True, 'SAeUron_ICML2025',
     'Sections 4.1–4.2; Section 7 explicitly discusses similar/abstract concepts, storage, removable runtime module.'),
    ('D15', 'SAEmnesia: Erasing Concepts in Diffusion Models with Supervised Sparse Autoencoders', 'ICML 2026 main', '2026-07', True, None,
     'Supervised SAE feature centralization and sequential multi-concept experiments.'),
    ('D16', 'One-dimensional Adapter to Rule Them All: Concepts, Diffusion Models and Erasing Applications', 'CVPR 2024 main', '2024-06', True, 'SPM_CVPR2024',
     'Sections 3.1–3.3: rank-one residual, latent anchoring, Facilitated Transport; related-model transfer.'),
    ('D17', 'SafeGuide: Adaptive Inference-Time Safety Control for Diffusion Models', 'ICLR 2026 Workshop: Principled Design for Trustworthy AI', '2026-04', True, 'SafeGuide_ICLR2026_Workshop',
     'Official accepted list and event abstract only. Bell-shaped schedule learned using risk-aware RL. PDF HTTP 403; no full-text-level claims.'),
    ('D21', 'Receler: Reliable Concept Erasing of Text-to-Image Diffusion Models via Lightweight Erasers', 'ECCV 2024 main', '2024-09/10', True, 'Receler_ECCV2024',
     'Section 3: lightweight residual erasers, concept-localized regularization, adversarial prompt learning; ECVA proceedings.'),
    ('X01', 'SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding', 'ACL 2024 main', '2024-08', True, 'SafeDecoding_ACL2024',
     'Sections 4.3–4.4: train safety expert using LoRA; intersect candidate sets and reweight token probabilities.'),
    ('X02', 'Please refuse to answer me! Mitigating Over-Refusal in Large Language Models via Adaptive Contrastive Decoding', 'ACL 2026 main', '2026-07', False, 'AdaCD_ACL2026',
     'Section 3: prompted/unprompted distribution difference, rank agreement and confidence-based mode switching.'),
    ('X03', 'HiddenGuard: Fine-Grained Safe Generation with Specialized Representation Router', 'ACL 2026 main', '2026-07', True, 'HiddenGuard_ACL2026',
     'Sections 4.1–4.3: learned low-rank signal heads and token-level router; base weights frozen.'),
    ('X04', 'Adaptive Backtracking for Privacy Protection in Large Language Models', 'Findings of ACL 2026', '2026-07', False, 'ABack_FindingsACL2026',
     'Sections 3.3–3.4: protected-entity monitor, lookahead, windowed reverse state inference, revision; prototypes precomputed using pretrained models.'),
    ('X05', 'A General Framework for Inference-time Scaling and Steering of Diffusion Models', 'ICML 2025 main', '2025-07', False, 'FKSteering_ICML2025',
     'Section 3: particle reweighting/resampling. Includes training-free implementations; optional learned intermediate rewards and gradient proposals are separate variants. T2I is already an application; toxicity experiments include increasing toxicity for red teaming.'),
    ('X06', 'Safe Control using Learned Safety Filters and Adaptive Conformal Inference', 'L4DC 2026', '2026-06-17/19', True, 'ACoFi_L4DC2026',
     'Section 3, Algorithm 1, Theorem 2: learned safety values/policy, adaptive uncertainty bounds; soft long-run guarantee with observed delayed targets.'),
    ('X07', 'Steering MoE LLMs via Expert (De)Activation', 'ICLR 2026 main', '2026-04', False, 'SteerMoE_ICLR2026',
     'Sections 3.1–3.2: paired expert-activation frequencies and inference-time router-score changes; architecture-dependent.'),
    ('X08', 'Disentangled Safety Adapters Enable Efficient Guardrails and Flexible Inference-Time Alignment', 'ICLR 2026 Workshop: Principled Design for Trustworthy AI', '2026-04', True, 'DSA_ICLR2026_Workshop',
     'Sections 3–4: side networks for classification and output-logit alignment; official workshop acceptance plus author version with workshop header. First preprint in 2025 is not its conference date.'),
    ('X09', 'Steering Safely or Off a Cliff? Rethinking Specificity and Robustness in Inference-Time Interventions', 'EACL 2026 main', '2026-03', None, 'SteeringSafely_EACL2026',
     'Evaluation-only inspiration: general/control/robustness specificity. Not proposed as a new erasure algorithm.'),
    ('X10', 'SteeringSafety: Benchmarking Representation Steering in LLMs Across Safety Perspectives', 'ICML 2026 main', '2026-07', None, 'SteeringSafety_ICML2026',
     'Evaluation-only inspiration: 9 safety perspectives and 18 datasets. Projection-based methods in benchmark not imported into the recommended algorithm set.'),
]

for key, title, venue, date, training, download_id, evidence in extra:
    rec = dict(id=key, title=title, venue=venue, conference_date=date,
               publication_status='accepted workshop' if 'Workshop' in venue else 'published',
               primary_url=links[key], new_parameter_training=training, evidence=evidence,
               verification_depth='official abstract only' if key == 'D17' else 'full text',
               scope_note='T2I representative' if key.startswith('D') else 'cross-domain or adjacent inspiration')
    if download_id:
        rec['download_id'] = download_id
        d = downloads.get(download_id, {})
        if d.get('status') == 'ok':
            rec['local_pdf'] = str(ROOT / d['local_pdf'])
            rec['local_text'] = str(ROOT / d['local_text'])
            rec['pdf_url'] = d['pdf_url']
    elif key == 'D11':
        rec['local_pdf'] = str(PAPERS / 'inference_no_projection/pdfs/2025_DAG.pdf')
    else:
        matches = [x for x in inventory if x.get('title') == title]
        if len(matches) != 1:
            raise ValueError(f'Expected unique local record for {key}: {title}')
        rec['local_pdf'] = str(PAPERS / matches[0]['filename'])
    if key in ['D07', 'D12']:
        rec['web_recheck_note'] = 'Page returned 403 on this run; formal CVF PDF watermark and previous bibliographic audit corroborate acceptance.'
    if key == 'D17':
        rec['acceptance_evidence'] = ['https://trustworthy-ai-workshop.github.io/iclr2026/papers', str(ROOT / 'sources/pages/SafeGuide_event.html')]
    if key == 'X08':
        rec['acceptance_evidence'] = ['https://trustworthy-ai-workshop.github.io/iclr2026/papers', links['X08A'], str(ROOT / 'sources/pages/DSA_event.html')]
        rec['pdf_version'] = 'public author version of accepted workshop paper'
    records.append(rec)

for rec in records:
    if rec.get('local_pdf'):
        p = Path(rec['local_pdf'])
        if not p.is_file():
            raise FileNotFoundError(p)
        rec['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
    rec['method_description_location'] = 'RESEARCH.zh-CN.md sections 1 and 3'

payload = {
    'checked_dates': ['2026-10-22', '2026-10-23'], 'timezone': 'Asia/Shanghai',
    'scope': 'Accepted works only. Frozen-backbone learned adapters are separately labeled. Excludes analytic projection/closed-form erasure operators from proposed routes.',
    'counts': {'papers': len(records), 'with_local_full_text': sum(bool(x.get('local_pdf')) for x in records),
               'abstract_only': sum(x['verification_depth'] == 'official abstract only' for x in records)},
    'records': sorted(records, key=lambda x: x['id']),
    'inspected_but_not_in_recommended_methods': [
        {'id': 'SafeSteer_FindingsACL2026', 'reason': 'Full method includes PCA-based probe construction; not selected as a non-projection transfer route.'},
        {'id': 'SafeCompass_ICML2026', 'reason': 'Useful dynamic monitoring prior, but uses a direction-based projection readout; omitted from main transfer list to respect the strict boundary. Its intervention is token insertion, not hidden-state projection erasure.'},
        {'id': 'SafeInfer_AAAI2025', 'reason': 'Requires separately fine-tuning an unsafe auxiliary LM; redundant relative to selected experts/side-network papers. Not training-free.'},
        {'title': 'Inference-time Unlearning Using Conformal Prediction', 'primary_url': 'https://research.google/pubs/inference-time-unlearning-using-conformal-prediction/',
         'reason': 'Primary author-institution page says to appear but no named accepted venue verified; excluded from accepted-paper results. Its existence is a reason not to claim that conformal inference-time unlearning is globally unexplored.'},
    ],
    'limitations': ['Representative selection, not exhaustive.', 'No new experiments run.', 'SafeGuide full text unavailable (HTTP 403).',
                    'Published and accepted works only cannot establish novelty against all current research.'],
}
(ROOT / 'sources/catalogue.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(payload['counts'], ensure_ascii=False))
