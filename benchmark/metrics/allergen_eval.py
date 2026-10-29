"""End-to-end allergen evaluation through real parser -> matcher -> allergen_engine."""
from __future__ import annotations

from typing import Any, Optional

from backend.allergen_engine import check_allergens
from backend.matcher import KnowledgeBase, match_ingredients
from backend.models import AllergenStatus, TranscriptionResult, UserProfile
from backend.parser import parse_ingredients


class AllergenEvaluator:
    """Evaluates transcription outputs against expected allergen profiles."""

    def __init__(self, kb: Optional[KnowledgeBase] = None):
        self.kb = kb or KnowledgeBase()

    def evaluate(
        self,
        transcription_data: dict[str, Any],
        expected_allergen_results: dict[str, str],
        engine_name: str = "benchmark_engine",
    ) -> dict[str, Any]:
        """Run transcription through parser -> matcher -> allergen engine and compare.

        Parameters
        ----------
        transcription_data:
            Dict containing ingredients_text, contains_statement, may_contain_statement, unreadable_parts.
        expected_allergen_results:
            Dict of {profile_name: expected_status_string} (e.g., {"milk": "CLEAR", "soy": "CONTAINS"}).
        engine_name:
            Identifier of engine being evaluated.

        Returns
        -------
        dict[str, Any]
            Detailed evaluation dictionary including false_clears, matches, confusion counts,
            and per-profile breakdown.
        """
        ingredients_text = transcription_data.get("ingredients_text") or ""
        contains_stmt = transcription_data.get("contains_statement") or ""
        may_contain_stmt = transcription_data.get("may_contain_statement") or ""
        unreadable = transcription_data.get("unreadable_parts") or []

        full_text = ingredients_text.strip()
        if contains_stmt and "contains:" not in full_text.lower():
            full_text = f"{full_text.rstrip('.')}. Contains: {contains_stmt.strip()}."
        if may_contain_stmt and "may contain" not in full_text.lower():
            full_text = f"{full_text.rstrip('.')}. May contain: {may_contain_stmt.strip()}."

        parsed = parse_ingredients(full_text)
        matches = match_ingredients(parsed.ingredients, self.kb)

        transcription_model = TranscriptionResult(
            ingredients_text=ingredients_text,
            contains_statement=contains_stmt,
            may_contain_statement=may_contain_stmt,
            unreadable_parts=unreadable,
            engine_used=engine_name,
            time_ms=transcription_data.get("timings", {}).get("elapsed_ms", 0),
            fallback_occurred=False,
        )

        confusion: dict[str, int] = {}
        per_profile: dict[str, dict[str, str]] = {}
        false_clears = 0
        total_checks = 0
        exact_matches = 0

        for profile_name, expected_raw in expected_allergen_results.items():
            expected = expected_raw.strip().upper()
            user_profile = UserProfile(allergens=[profile_name])

            result = check_allergens(
                matches=matches,
                profile=user_profile,
                parsed=parsed,
                transcription=transcription_model,
            )

            actual = result.overall_status.value.upper()
            total_checks += 1

            if actual == expected:
                exact_matches += 1

            pair_key = f"{expected}->{actual}"
            confusion[pair_key] = confusion.get(pair_key, 0) + 1

            # Check FALSE_CLEAR violation: expected CONTAINS or MIGHT_CONTAIN, got CLEAR
            is_fc = (expected in ("CONTAINS", "MIGHT_CONTAIN")) and (actual == "CLEAR")
            if is_fc:
                false_clears += 1

            per_profile[profile_name] = {
                "expected": expected,
                "actual": actual,
                "is_match": actual == expected,
                "is_false_clear": is_fc,
            }

        allergen_accuracy = round(exact_matches / total_checks, 4) if total_checks > 0 else 1.0

        return {
            "total_checks": total_checks,
            "exact_matches": exact_matches,
            "allergen_accuracy": allergen_accuracy,
            "false_clears": false_clears,
            "confusion_counts": confusion,
            "per_profile": per_profile,
        }
