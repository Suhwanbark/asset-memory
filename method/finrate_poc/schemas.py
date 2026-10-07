"""Schemas constrain serialization, never gold answers or desired outcomes."""


def obj(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


STRING = {"type": "string"}
FACT = obj({"subject": STRING, "topic": STRING, "predicate": STRING,
            "value": STRING, "period": {"type": ["string", "null"]},
            "modality": {"type": "string", "enum": ["reported", "risk", "forecast", "plan", "explicit_retraction"]},
            "source_id": STRING, "quote": STRING})
EXTRACTION = obj({"facts": {"type": "array", "items": FACT}})
LINK = obj({"old_id": STRING, "new_id": STRING,
            "relation": {"type": "string", "enum": ["continues", "updates", "contradicts", "retracts", "same_topic"]},
            "rationale": STRING, "evidence_quote": STRING})
LINKS = obj({"links": {"type": "array", "items": LINK}})
JUDGEMENT = obj({
    "label": {"type": "string", "enum": ["CORRECT", "PARTIAL", "INCORRECT", "FAILURE"]},
    "key_points_met": {"type": "array", "items": {"type": "boolean"}},
    "citation_support": {"type": "string", "enum": ["supported", "partial", "unsupported", "no_citations"]},
    "temporal_error": {"type": "boolean"}, "entity_error": {"type": "boolean"}, "reason": STRING})
