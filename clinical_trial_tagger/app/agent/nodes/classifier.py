from app.agent.state import AgentState


def classifier_node(state: AgentState) -> dict:
    try:
        retrieved_chunks = state.get("retrieved_chunks") or []

        vote_counts: dict[str, int] = {}
        for chunk in retrieved_chunks:
            category = chunk.get("category")
            if category:
                vote_counts[category] = vote_counts.get(category, 0) + 1

        if not vote_counts:
            return {"vote_counts": {}, "confidence": 0.0, "predicted_category": ""}

        total_votes = sum(vote_counts.values())
        predicted_category = max(vote_counts, key=vote_counts.get)
        confidence = vote_counts[predicted_category] / total_votes

        return {
            "vote_counts": vote_counts,
            "confidence": confidence,
            "predicted_category": predicted_category,
        }
    except Exception as exc:
        return {
            "error": f"classifier_node failed: {exc}",
            "vote_counts": {},
            "confidence": 0.0,
            "predicted_category": "",
        }
