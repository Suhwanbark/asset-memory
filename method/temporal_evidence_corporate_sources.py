"""Archive six fixed corporate-event histories before model inference.

Full public issuer/regulator documents, never claim-selected excerpts. Restricted
SEC/FTC/DOJ endpoints are not bypassed; alternate first-party publications and
the resulting change in scope are explicitly recorded. No market linkage claim.
"""
from __future__ import annotations

import concurrent.futures
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re

from bs4 import BeautifulSoup
from pypdf import PdfReader
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/temporal_evidence_poc_v1"


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def source(sid, url, date, title, selector, minimum, required, kind="press_release"):
    return dict(id=sid, url=url, publication_date=date, title=title,
                selector=selector, minimum_characters=minimum,
                required_phrases=required, kind=kind)


SOURCES = [
    source("corp_jblu_20230307", "https://news.jetblue.com/latest-news/press-release-details/2023/JetBlue-and-Spirit-Will-Continue-to-Advance-Plan-to-Create-Compelling-National-Low-Fare-Challenger-to-the-Dominant-U.S.-Carriers/default.aspx", "2023-03-07", "JetBlue and Spirit Will Continue to Advance Plan", "main#maincontent", 10500, ["JetBlue and Spirit will continue to advance", "About Spirit"]),
    source("corp_jblu_20240304", "https://news.jetblue.com/latest-news/press-release-details/2024/JetBlue-Announces-Termination-of-Merger-Agreement-with-Spirit/", "2024-03-04", "JetBlue Announces Termination of Merger Agreement with Spirit", "main#maincontent", 8000, ["terminate their July 2022 merger agreement", "Forward"]),
    source("corp_adbe_20231128", "https://www.gov.uk/government/news/adobe-figma-deal-could-harm-uk-digital-design-sector", "2023-11-28", "Adobe / Figma deal could harm UK digital design sector", "main#content", 6000, ["Today’s decision is provisional", "Notes to editors"]),
    source("corp_adbe_20231218", "https://news.adobe.com/news/news-details/2023/adobe-and-figma-mutually-agree-to-terminate-merger-agreement", "2023-12-18", "Adobe and Figma Mutually Agree to Terminate Merger Agreement", "main", 6000, ["mutual agreement to terminate", "Forward"]),
    source("corp_irbt_20230725", "https://press.aboutamazon.com/2023/7/amazon-and-irobot-modify-merger-price", "2023-07-25", "Amazon and iRobot Modify Merger Price", "article.PressReleasePage-mainContent", 13000, ["$51.75 per share", "Forward"]),
    source("corp_irbt_20240129", "https://www.aboutamazon.com/news/company-news/amazon-irobot-terminate-acquisition", "2024-01-29", "Amazon and iRobot agree to terminate pending acquisition", "article.ArticlePage-mainContent", 2400, ["mutual agreement to terminate", "Colin Angle"]),
    source("corp_atvi_20230426", "https://www.gov.uk/government/news/microsoft-activision-deal-prevented-to-protect-innovation-and-choice-in-cloud-gaming", "2023-04-26", "Microsoft / Activision deal prevented to protect innovation and choice in cloud gaming", "main#content", 7300, ["The CMA has prevented", "Notes to editors"]),
    source("corp_atvi_20231013", "https://www.gov.uk/government/news/microsoft-concession-a-gamechanger-that-will-promote-competition", "2023-10-13", "Microsoft concession a gamechanger that will promote competition", "main#content", 4500, ["clearing this narrower transaction", "outside the EEA"]),
    source("corp_kr_20240226", "https://ir.kroger.com/news/news-details/2024/Kroger-Issues-Statement-on-FTC-Decision-to-Block-Proposed-Merger-with-Albertsons-Companies/default.aspx", "2024-02-26", "Kroger Issues Statement on FTC Decision to Block Proposed Merger with Albertsons Companies", ".evergreen-news-details-item", 8800, ["proposed merger", "Forward"]),
    source("corp_aci_20241211", "https://albertsons2021rd.q4web.com/newsroom/press-releases/news-details/2024/Albertsons-Terminates-Merger-Agreement/default.aspx", "2024-12-11", "Albertsons Terminates Merger Agreement", "#maincontent", 11500, ["exercised its right to terminate", "Forward"]),
    source("corp_vaneck_20231207", "https://www.vaneck.com/us/en/blogs/digital-assets/matthew-sigel-vanecks-15-crypto-predictions-for-2024/", "2023-12-07", "VanEck’s 15 Crypto Predictions for 2024", "div.blog-post.content", 23500, ["first spot Bitcoin ETFs", "2024"], kind="issuer_research_blog"),
    source("corp_vaneck_20240111", "https://www.vaneck.com/us/en/press-releases/time-to-hodl-vaneck-launches-spot-bitcoin-etf.pdf", "2024-01-11", "Time to HODL: VanEck Launches VanEck Bitcoin Trust", None, 7000, ["January 11, 2024", "HODL", "risk"], kind="issuer_press_release_pdf"),
]

ACCESS_EXCLUSIONS = [
    dict(url="https://www.justice.gov/archives/opa/pr/justice-department-statements-jetblue-terminating-acquisition-spirit-airlines", reason="HTTP 200 but 2538-byte challenge/no main document", replacement="corp_jblu_20240304", note="Same event and publication day, company release instead of DOJ statement."),
    dict(url="https://www.ftc.gov/news-events/news/press-releases/2024/02/ftc-challenges-krogers-acquisition-albertsons", reason="HTTP 403", replacement="corp_kr_20240226", note="Issuer response, not the FTC release; corporate opinions are not labeled regulatory findings."),
    dict(url="https://www.ftc.gov/news-events/news/press-releases/2024/12/statement-ftc-victory-securing-halt-kroger-albertsons-grocery-merger", reason="HTTP 403", replacement=None, note="Intermediate injunction release excluded; no attempt to reconstruct it."),
    dict(url="https://www.sec.gov/Archives/edgar/data/56873/000110465924127669/tm2427516d10_ex99-1.htm", reason="HTTP 403 undeclared automated tool", replacement="corp_aci_20241211", note="Public Albertsons issuer termination release instead of Kroger's SEC-filed announcement; termination-fee dispute is not adjudicated."),
    dict(url="https://www.sec.gov/newsroom/speeches-statements/peirce-uyeda-statement-vaneck-bitcoin-trust-031023", reason="HTTP 403 request rate threshold", replacement="corp_vaneck_20231207", note="Scope changed: dated issuer expectation of US spot-ETF debut, not March legal disapproval."),
    dict(url="https://www.sec.gov/newsroom/speeches-statements/gensler-statement-spot-bitcoin-011023", reason="HTTP 403 request rate threshold", replacement="corp_vaneck_20240111", note="Scope changed: actual issuer launch, not independent SEC legal approval/endorsement."),
]


def archive(spec):
    folder = OUT / "sources" / spec["id"]
    meta_file = folder / "meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        assert meta["url"] == spec["url"]
        assert sha(Path(meta["text_path"]).read_bytes()) == meta["text_sha256"]
        assert sha(Path(meta["raw_path"]).read_bytes()) == meta["raw_sha256"]
        return meta
    response = requests.get(spec["url"], timeout=45)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    page_count = None
    if spec["kind"].endswith("_pdf"):
        if not response.content.startswith(b"%PDF-"):
            raise ValueError(f"Not PDF: {spec['id']}")
        reader = PdfReader(io.BytesIO(response.content))
        page_texts = [page.extract_text() or "" for page in reader.pages]
        if any(len(t.strip()) < 80 for t in page_texts):
            raise ValueError(f"Unextracted/empty PDF page: {spec['id']}")
        raw_text = "\n\n".join(page_texts)
        page_count = len(page_texts)
        extension = "pdf"
    else:
        if "html" not in content_type:
            raise ValueError(f"Not HTML: {spec['id']} ({content_type})")
        soup = BeautifulSoup(response.content, "html.parser")
        body = soup.select_one(spec["selector"])
        if body is None:
            raise ValueError(f"Missing full-document selector: {spec['id']}")
        for node in body.select("script,style,svg,noscript"):
            node.decompose()
        raw_text = body.get_text("\n", strip=True)
        extension = "html"
    # Whitespace-only normalization. All alphanumeric source characters survive.
    text = raw_text.replace("\xa0", " ").strip() + "\n"
    original_alnum = "".join(c for c in raw_text if c.isalnum())
    output_alnum = "".join(c for c in text if c.isalnum())
    assert original_alnum == output_alnum
    flat = re.sub(r"\s+", " ", text)
    if len(text) < spec["minimum_characters"]:
        raise ValueError(f"Short/challenge source: {spec['id']} {len(text)}")
    if any(marker in flat.lower() for marker in ["your request originates from an undeclared automated tool", "request rate threshold exceeded", "verify you are human"]):
        raise ValueError(f"Access challenge: {spec['id']}")
    for phrase in spec["required_phrases"]:
        if phrase.casefold() not in flat.casefold():
            raise ValueError(f"Missing completeness anchor {phrase!r}: {spec['id']}")
    folder.mkdir(parents=True, exist_ok=True)
    raw_path = folder / f"source.{extension}"
    text_path = folder / "source.txt"
    raw_path.write_bytes(response.content)
    text_path.write_text(text)
    meta = {k: spec[k] for k in ["id", "url", "publication_date", "title", "kind"]}
    meta.update(text_path=str(text_path), raw_path=str(raw_path),
                text_sha256=sha(text), raw_sha256=sha(response.content),
                characters=len(text), raw_bytes=len(response.content),
                http_status=response.status_code, response_url=response.url,
                content_type=content_type, main_selector=spec["selector"],
                pdf_pages=page_count, all_pages_extracted=(page_count is not None),
                source_alphanumeric_sha256=sha(original_alnum),
                all_main_document_alphanumeric_characters_preserved=True,
                completeness_anchors=spec["required_phrases"],
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                scope="entire publisher main release/article, or every PDF page; no claim-based paragraph selection")
    write_json(meta_file, meta)
    return meta


# Gold specs are static, committed in this source before any downstream model run.
SPECS = [
    dict(id="corporate_jetblue_spirit", title="JetBlue-Spirit merger termination", event_date="2024-03-04", cutoff="2024-03-05T23:59:59Z", docs=["corp_jblu_20230307", "corp_jblu_20240304"], probes=[
        ("pending", "current", "JetBlue and Spirit are still pursuing completion of their July 2022 merger agreement.", "contradicted", "superseded_pending_status", ["corp_jblu_20240304"], "The March 4 release says the parties agreed to terminate the agreement."),
        ("history", "historical", "On March 7, 2023, JetBlue and Spirit stated that they would continue to pursue their proposed merger despite the DOJ complaint.", "supported", "historical_record", ["corp_jblu_20230307"], "The dated issuer statement records that intention, not actual approval."),
        ("new", "current", "JetBlue and Spirit have agreed to terminate their July 2022 merger agreement.", "supported", "current_event_status", ["corp_jblu_20240304"], "Termination is explicitly announced in the latest supplied release.")]),
    dict(id="corporate_adobe_figma", title="Adobe-Figma merger termination", event_date="2023-12-18", cutoff="2023-12-19T23:59:59Z", docs=["corp_adbe_20231128", "corp_adbe_20231218"], probes=[
        ("pending", "current", "Adobe and Figma are still pursuing completion of the merger agreement while awaiting the CMA's scheduled final review decision.", "contradicted", "superseded_pending_status", ["corp_adbe_20231218"], "The December 18 release announces mutual termination before that review deadline."),
        ("history", "historical", "On November 28, 2023, the CMA's published competition findings on the Adobe-Figma deal were provisional.", "supported", "historical_record", ["corp_adbe_20231128"], "The regulator explicitly called its findings provisional."),
        ("new", "current", "Adobe and Figma have mutually agreed to terminate their previously announced merger agreement.", "supported", "current_event_status", ["corp_adbe_20231218"], "Explicit issuer announcement; not a claim that the provisional CMA finding became a final prohibition.")]),
    dict(id="corporate_amazon_irobot", title="Amazon-iRobot acquisition termination", event_date="2024-01-29", cutoff="2024-01-30T23:59:59Z", docs=["corp_irbt_20230725", "corp_irbt_20240129"], probes=[
        ("pending", "current", "The amended Amazon-iRobot acquisition remains pending under terms of $51.75 per iRobot share, subject to its closing conditions.", "contradicted", "superseded_pending_status", ["corp_irbt_20240129"], "The subsequent issuer announcement terminates the acquisition agreement."),
        ("history", "historical", "The amended merger terms announced on July 25, 2023 provided for Amazon to pay $51.75 per iRobot share.", "supported", "historical_record", ["corp_irbt_20230725"], "The offer consideration is explicitly stated in the historical release."),
        ("new", "current", "Amazon and iRobot have mutually agreed to terminate their acquisition agreement.", "supported", "current_event_status", ["corp_irbt_20240129"], "The latest issuer release directly states termination.")]),
    dict(id="corporate_microsoft_activision", title="CMA clearance of narrower Microsoft-Activision transaction", event_date="2023-10-13", cutoff="2023-10-14T23:59:59Z", docs=["corp_atvi_20230426", "corp_atvi_20231013"], probes=[
        ("pending", "current", "The CMA still has not cleared a transaction allowing Microsoft to acquire Activision.", "contradicted", "superseded_overbroad_status", ["corp_atvi_20231013"], "The CMA cleared a narrower transaction; the old original-scope block cannot support this blanket current assertion."),
        ("history", "historical", "On April 26, 2023, the CMA prevented Microsoft's original proposed purchase of Activision over cloud-gaming competition concerns.", "supported", "historical_record", ["corp_atvi_20230426"], "The original broader proposal was blocked at that time."),
        ("new", "current", "The CMA has cleared the narrower Microsoft-Activision transaction after the concession placing specified cloud-streaming rights outside the EEA with Ubisoft.", "supported", "current_scoped_status", ["corp_atvi_20231013"], "Approval depends on narrower scope; it is not approval of the unchanged original transaction.")]),
    dict(id="corporate_kroger_albertsons", title="Kroger-Albertsons merger termination", event_date="2024-12-11", cutoff="2024-12-12T23:59:59Z", docs=["corp_kr_20240226", "corp_aci_20241211"], probes=[
        ("pending", "current", "Albertsons continues to pursue completion of its proposed merger with Kroger under the existing merger agreement.", "contradicted", "superseded_pending_status", ["corp_aci_20241211"], "Albertsons announced that it had exercised its right to terminate the agreement."),
        ("history", "historical", "On February 26, 2024, Kroger publicly defended its proposed Albertsons merger in response to the FTC's challenge.", "supported", "historical_record", ["corp_kr_20240226"], "The historical document is Kroger's dated response, not a court judgment."),
        ("new", "current", "Albertsons has announced that it exercised its right to terminate its merger agreement with Kroger following court injunctions.", "supported", "current_issuer_reported_status", ["corp_aci_20241211"], "This matches the issuer's announcement without deciding the parties' termination-fee dispute.")]),
    dict(id="corporate_vaneck_spot_bitcoin", title="US spot bitcoin ETP debut: expectation to issuer launch", event_date="2024-01-11", cutoff="2024-01-12T23:59:59Z", docs=["corp_vaneck_20231207", "corp_vaneck_20240111"], scope_change="Issuer expectation-to-launch, not SEC legal approval. The original SEC disapproval/approval statements were access-restricted and excluded.", probes=[
        ("pending", "current", "The debut of US-listed spot bitcoin exchange-traded products is still only a future event, and VanEck has not launched a US spot bitcoin product.", "contradicted", "superseded_prelaunch_status", ["corp_vaneck_20240111"], "The January 11 issuer release announces the launch of HODL, providing spot bitcoin exposure."),
        ("history", "historical", "In its December 7, 2023 predictions, VanEck expected the first US spot bitcoin ETFs to arrive in 2024.", "supported", "historical_forecast_record", ["corp_vaneck_20231207"], "This is a claim about an actual dated forecast, not a claim that every forecast in the article was realized."),
        ("new", "current", "VanEck has announced the launch of the VanEck Bitcoin Trust under ticker HODL as an exchange-traded product providing spot bitcoin exposure.", "supported", "current_product_launch", ["corp_vaneck_20240111"], "The January 11 first-party launch release supports product availability, not investment endorsement or a guaranteed return.")]),
]


def make_case(spec, sources):
    docs = [sources[sid] for sid in spec["docs"]]
    docs.sort(key=lambda x: (x["publication_date"], x["id"]))
    probes, labels = [], []
    for i, row in enumerate(sorted(spec["probes"], key=lambda row: sha(spec["id"] + row[0]))):
        name, scope, claim, label, category, evidence, explanation = row
        pid = f"p{i+1}"
        probes.append(dict(id=pid, scope=scope, claim=claim))
        labels.append(dict(case_id=spec["id"], probe_id=pid, label=label,
                           category=category, source_ids=evidence, explanation=explanation))
    case = dict(id=spec["id"], title=spec["title"], event_date=spec["event_date"],
                cutoff=spec["cutoff"], source_stratum="official_event_not_market_linked",
                event_family="corporate_event", market_reference=None,
                documents=docs, probes=probes,
                inclusion="Six purposive corporate-event histories fixed before model inference; no model-outcome-based selection.",
                dependence="Independent named events; merger termination dominates this small diagnostic stratum.")
    if "scope_change" in spec:
        case["scope_change"] = spec["scope_change"]
    assert all(d["publication_date"] <= spec["cutoff"][:10] for d in docs)
    return case, labels


def main():
    sources, failures = {}, []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(archive, spec): spec for spec in SOURCES}
        for future in concurrent.futures.as_completed(futures):
            spec = futures[future]
            try:
                doc = future.result()
                sources[doc["id"]] = doc
                print("ARCHIVED", doc["id"], doc["characters"], flush=True)
            except Exception as exc:
                failures.append(dict(source_id=spec["id"], url=spec["url"], reason=str(exc)))
                print("EXCLUDED", spec["id"], str(exc), flush=True)
    cases, gold, excluded = [], [], []
    for spec in SPECS:
        missing = [sid for sid in spec["docs"] if sid not in sources]
        if missing:
            excluded.append(dict(event_id=spec["id"], missing_source_ids=missing))
            continue
        case, labels = make_case(spec, sources)
        cases.append(case)
        gold.extend(labels)
    timestamp = datetime.now(timezone.utc).isoformat()
    write_json(OUT / "corporate_cases.json", dict(version="v1", built_at=timestamp,
        cases=cases, exclusions=excluded, source_failures=failures,
        access_restricted_sources_excluded=ACCESS_EXCLUSIONS,
        full_sources_archived=True, prediction_market_contracts_verified=False))
    write_json(OUT / "corporate_gold.json", dict(version="v1", fixed_before_inference_at=timestamp,
        construction_script_sha256=sha(Path(__file__).read_bytes()),
        labels=gold, reference="Explicit source-backed event state, historical record, and document scope; never asset prices or later returns."))
    print("BUILT", len(cases), "events", len(gold), "probes", "EXCLUDED", len(excluded), flush=True)


if __name__ == "__main__":
    main()
