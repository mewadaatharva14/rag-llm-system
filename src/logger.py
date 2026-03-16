"""
Query Logger
=============
Logs every query with full details to a JSON file.

Each log entry contains:
  timestamp, question, route_used, cosine_score,
  rerank_score, answer, source_documents, latency_ms

Why logging matters:
  Shows observability thinking — understanding what your
  system does in production. Recruiters rarely see this
  in student projects. The log file makes the routing
  decisions visible and debuggable.
"""

import json
import os
import time
from datetime import datetime
from typing import List, Dict


class QueryLogger:
    def __init__(self, config: dict):
        self.log_file = config["logging"]["log_file"]
        self.max_logs = config["logging"]["max_logs"]
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def log(
        self,
        question:        str,
        answer:          str,
        route:           str,
        reason:          str,
        cosine_score:    float,
        rerank_score:    float,
        source_chunks:   List[Dict],
        latency_ms:      float,
    ) -> None:
        """
        Append one query log entry to the JSON log file.
        Rotates log if max_logs exceeded.
        """
        entry = {
            "timestamp":        datetime.utcnow().isoformat(),
            "question":         question,
            "answer":           answer,
            "route_used":       route,
            "routing_reason":   reason,
            "cosine_score":     round(cosine_score, 4),
            "rerank_score":     round(rerank_score, 4) if rerank_score else None,
            "latency_ms":       round(latency_ms, 2),
            "source_documents": list({c["source"] for c in source_chunks}),
            "num_chunks_used":  len(source_chunks),
        }

        logs = self._load_logs()
        logs.append(entry)

        # rotate if over limit
        if len(logs) > self.max_logs:
            logs = logs[-self.max_logs:]

        with open(self.log_file, "w") as f:
            json.dump(logs, f, indent=2)

    def _load_logs(self) -> List[Dict]:
        if not os.path.exists(self.log_file):
            return []
        try:
            with open(self.log_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

    def get_recent(self, n: int = 20) -> List[Dict]:
        """Return the n most recent log entries."""
        logs = self._load_logs()
        return logs[-n:]

    def get_stats(self) -> Dict:
        """Return summary stats — route distribution, avg scores."""
        logs = self._load_logs()
        if not logs:
            return {"total_queries": 0}

        routes     = [l["route_used"] for l in logs]
        rag_count  = routes.count("rag") + routes.count("reranked")
        llm_count  = routes.count("llm")
        avg_cosine = sum(l["cosine_score"] for l in logs) / len(logs)
        avg_lat    = sum(l["latency_ms"] for l in logs) / len(logs)

        return {
            "total_queries":  len(logs),
            "rag_route":      rag_count,
            "llm_route":      llm_count,
            "avg_cosine":     round(avg_cosine, 4),
            "avg_latency_ms": round(avg_lat, 2),
        }