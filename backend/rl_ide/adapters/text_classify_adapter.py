"""Text Classification & Ticket Triage Domain Adapter.

Implements the DomainAdapter protocol for natural language support ticket
and email triage routing:
    Actions: ["urgent", "normal", "spam", "escalate"]
"""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from rl_ide.contracts.adapter import (
    ObservationSpec,
    ActionSpec,
    StepResult,
    DomainAdapter,
)


def generate_synthetic_tickets(n_records: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generates synthetic customer support tickets and emails with rich text and metadata.

    Design Invariants:
    - VIP customers with high angry tone or urgency keywords must be escalated.
    - Security / Outage / Down issues must be marked urgent.
    - Promotional, lottery, gift card, or pharma patterns are spam.
    - Routine questions (password reset, billing receipt) are normal.
    """
    rng = np.random.default_rng(seed)

    vip_subjects = [
        "PRODUCTION OUTAGE: API gateway returning 502 errors",
        "CRITICAL: Database primary node unresponsive",
        "Payment processing down - losing transactions immediately",
        "Executive escalation: Contract SLA breach on uptime",
        "Disaster recovery restore request for cluster #9",
        "Urgent: Security vulnerability report from our audit team",
        "Billing error: Double charged $24,000 on monthly invoice",
        "Cannot access dedicated enterprise tenant console",
    ]

    vip_bodies = [
        "Our entire production cluster went offline 10 minutes ago. We need an on-call engineer ASAP. This is completely unacceptable for an enterprise tier customer.",
        "Transactions are dropping across all European regions. Our CEO is demanding a status update within the hour. Please escalate to engineering immediately.",
        "Our automated failover failed to trigger. We require immediate root-cause investigation and executive escalation.",
        "Your latest release broke our SSO SAML integration. 500 employees cannot log in to start their shift. Fix this now.",
    ]

    normal_subjects = [
        "How do I update my billing credit card?",
        "Question regarding user seat allocation in workspace",
        "Feature request: Export reports to Parquet format",
        "Request for receipt for invoice #INV-88219",
        "How to configure webhook retry policy?",
        "Updating primary email address on account",
        "Documentation clarification for pagination parameters",
        "Scheduled maintenance window inquiry",
        "Assistance with migrating from API v1 to v2",
        "Password reset link expired, please resend",
    ]

    normal_bodies = [
        "Hi support team, I would like to update the credit card on file for our team subscription. Where can I find the billing tab in settings? Thanks!",
        "Could you please send me a PDF receipt for our recent subscription renewal? Our accounting department requires it for tax filing.",
        "Hello, we noticed that webhook deliveries have a 5-second timeout. Is there any way to adjust this to 10 seconds? Best regards.",
        "We are planning our quarterly update and want to know if there is any scheduled downtime next weekend. Thank you for your assistance.",
        "I was trying to reset my password earlier today, but by the time I checked my inbox the link had expired. Could you help trigger a new one? Thanks.",
    ]

    spam_subjects = [
        "CONGRATULATIONS! You have won a $1,000 Amazon Gift Card!",
        "Claim your exclusive crypto reward: 2.5 ETH waiting",
        "Lowest rates on prescription medications online - no prescription required",
        "Urgent business proposal: Transfer of $14.5M USD funds",
        "Double your investment in 48 hours guaranteed",
        "Work from home and earn $5,000 weekly - click here",
        "Hot singles in your area want to chat now",
        "Account Suspended: Verify your Apple ID immediately or lose access",
    ]

    spam_bodies = [
        "Dear Winner, your email was randomly selected to receive a $1,000 gift card. Click the secure link below to verify your claim before midnight!",
        "Exclusive opportunity for selected investors. Deposit 0.1 BTC today and receive 100% daily returns guaranteed by smart contracts.",
        "Special discount: Buy high quality medications online with overnight delivery. 100% discreet packaging and no doctor visit needed.",
        "I am Barrister David representing a deceased client with $14.5M in dormant bank accounts. Contact me immediately to facilitate fund transfer.",
    ]

    urgent_subjects = [
        "URGENT: SSL certificate expired on login portal",
        "Outage detected: Webhook dispatch queue backed up by 50,000 events",
        "API rate limiting triggering for all internal services",
        "Security Alert: Multiple failed admin logins from unknown IP",
        "Critical bug: Data export returning empty files",
        "Urgent: Webhooks delivering duplicate charges to end-users",
    ]

    urgent_bodies = [
        "Our monitoring alerts indicate the main SSL certificate expired 5 minutes ago. Users are seeing browser security warnings and cannot authenticate.",
        "Webhook queue has exceeded 50k items and latency is spiking past 30 seconds. We need infrastructure support to scale consumer workers immediately.",
        "Internal services are receiving 429 Too Many Requests errors. Service-to-service calls should not be throttled. Please unblock urgently.",
        "Multiple unauthorized login attempts detected from Tor exit nodes on superadmin accounts. Immediate IP blacklisting required.",
    ]

    ticket_ids = []
    subjects = []
    bodies = []
    sender_tiers = []
    sender_is_vips = []
    sender_is_internals = []
    word_counts = []
    urgency_keyword_counts = []
    angry_tone_scores = []
    has_attachments = []
    queue_loads = []
    true_labels = []

    urgency_words = {"urgent", "outage", "down", "critical", "immediately", "asap", "failure", "emergency", "broken", "breach"}

    for i in range(n_records):
        t_id = f"TICK-{1000 + i}"
        ticket_ids.append(t_id)

        # Decide category distribution:
        # ~20% VIP/Escalate, ~20% Urgent, ~20% Spam, ~40% Normal
        roll = rng.random()

        if roll < 0.20:
            # VIP / Escalate category
            subj = str(rng.choice(vip_subjects))
            body = str(rng.choice(vip_bodies))
            tier = "vip"
            is_vip = True
            is_internal = False
            angry = float(rng.uniform(0.65, 0.98))
            has_att = bool(rng.random() > 0.6)
            q_load = int(rng.integers(10, 45))
            label = "escalate"

        elif roll < 0.40:
            # Urgent technical issue
            subj = str(rng.choice(urgent_subjects))
            body = str(rng.choice(urgent_bodies))
            tier = "internal" if rng.random() > 0.4 else "free"
            is_vip = False
            is_internal = (tier == "internal")
            angry = float(rng.uniform(0.3, 0.7))
            has_att = bool(rng.random() > 0.4)
            q_load = int(rng.integers(15, 50))
            label = "urgent"

        elif roll < 0.60:
            # Spam / Phishing
            subj = str(rng.choice(spam_subjects))
            body = str(rng.choice(spam_bodies))
            tier = "free"
            is_vip = False
            is_internal = False
            angry = float(rng.uniform(0.0, 0.2))
            has_att = bool(rng.random() > 0.7)
            q_load = int(rng.integers(5, 30))
            label = "spam"

        else:
            # Normal support request
            subj = str(rng.choice(normal_subjects))
            body = str(rng.choice(normal_bodies))
            tier = "free" if rng.random() > 0.2 else "vip"
            is_vip = (tier == "vip")
            is_internal = False
            angry = float(rng.uniform(0.0, 0.25))
            has_att = bool(rng.random() > 0.8)
            q_load = int(rng.integers(5, 35))
            label = "normal"

        full_text = f"{subj} {body}".lower()
        w_count = len(full_text.split())
        u_count = sum(1 for w in urgency_words if w in full_text)

        subjects.append(subj)
        bodies.append(body)
        sender_tiers.append(tier)
        sender_is_vips.append(is_vip)
        sender_is_internals.append(is_internal)
        word_counts.append(w_count)
        urgency_keyword_counts.append(u_count)
        angry_tone_scores.append(round(angry, 3))
        has_attachments.append(has_att)
        queue_loads.append(q_load)
        true_labels.append(label)

    # Anomaly flag for binary/conformance compatibility (True for non-normal)
    true_anomalies = [lbl in ("urgent", "spam", "escalate") for lbl in true_labels]

    df = pd.DataFrame({
        "ticket_id": ticket_ids,
        "subject": subjects,
        "body": bodies,
        "sender_tier": sender_tiers,
        "sender_is_vip": sender_is_vips,
        "sender_is_internal": sender_is_internals,
        "word_count": word_counts,
        "urgency_keyword_count": urgency_keyword_counts,
        "angry_tone_score": angry_tone_scores,
        "has_attachment": has_attachments,
        "queue_load": queue_loads,
        "true_label": true_labels,
        "true_anomaly": true_anomalies,
    })

    return df


class TextClassifyAdapter:
    """DomainAdapter for NLP customer support ticket and email triage routing.
    
    Actions:
        0: "urgent"
        1: "normal"
        2: "spam"
        3: "escalate"
    """

    ACTION_LABELS = ["urgent", "normal", "spam", "escalate"]
    EMBEDDING_DIM = 64
    METADATA_DIM = 7
    TOTAL_DIM = EMBEDDING_DIM + METADATA_DIM  # 71

    FEATURE_NAMES = (
        [f"text_emb_{i}" for i in range(EMBEDDING_DIM)]
        + [
            "sender_is_vip",
            "sender_is_internal",
            "word_count_norm",
            "urgency_keyword_norm",
            "angry_tone_score",
            "has_attachment",
            "queue_load_norm",
        ]
    )

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        episode_length: int = 25,
    ):
        self.df = df if df is not None else generate_synthetic_tickets(n_records=500, seed=42)
        self.episode_length = episode_length

        self.episode_id = 0
        self.current_idx = 0
        self.step_in_episode = 0
        self.processed_count = 0

        # Fast deterministic 64-dim HashingVectorizer
        self._vectorizer = HashingVectorizer(
            n_features=self.EMBEDDING_DIM,
            alternate_sign=False,
            norm="l2",
        )

        # Precompute text embeddings across dataset for ultra-fast step execution (<10µs)
        full_texts = [
            f"{s} {b}" for s, b in zip(self.df["subject"], self.df["body"])
        ]
        self._text_embeddings = self._vectorizer.transform(full_texts).toarray().astype(np.float32)

    def observations(self) -> ObservationSpec:
        return ObservationSpec(
            vector_dim=self.TOTAL_DIM,
            feature_names=self.FEATURE_NAMES,
            horizon_type="contextual_bandit",
            default_gamma=0.0,
        )

    def actions(self) -> ActionSpec:
        return ActionSpec(
            action_type="discrete",
            labels=self.ACTION_LABELS,
        )

    def event_schema(self) -> dict[str, str]:
        """Returns {field_path: type_str} legal in Reward DSL for ticket triage."""
        return {
            "obs.sender_is_vip": "bool",
            "obs.sender_is_internal": "bool",
            "obs.sender_tier": "str",
            "obs.word_count": "int",
            "obs.urgency_keyword_count": "int",
            "obs.angry_tone_score": "float",
            "obs.has_attachment": "bool",
            "obs.queue_load": "int",
            "action.label": "str",
            "action.action_idx": "int",
            "state.queue_load": "int",
            "state.processed_count": "int",
            "truth.has_label": "bool",
            "truth.label": "str",
            "truth.true_anomaly": "bool",
            "truth.is_vip": "bool",
        }

    def _get_obs_vector(self, row_idx: int) -> list[float]:
        row = self.df.iloc[row_idx]
        text_emb = self._text_embeddings[row_idx].tolist()

        metadata = [
            1.0 if row["sender_is_vip"] else 0.0,
            1.0 if row["sender_is_internal"] else 0.0,
            float(min(row["word_count"] / 100.0, 2.0)),
            float(min(row["urgency_keyword_count"] / 5.0, 2.0)),
            float(row["angry_tone_score"]),
            1.0 if row["has_attachment"] else 0.0,
            float(min(row["queue_load"] / 50.0, 1.0)),
        ]

        return text_emb + metadata

    def reset(self, seed: int | None = None) -> tuple[list[float], dict[str, Any]]:
        self.step_in_episode = 0
        self.processed_count = 0

        if seed is not None:
            rng = np.random.default_rng(seed)
            max_start = max(0, len(self.df) - self.episode_length - 1)
            self.current_idx = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        else:
            max_start = max(1, len(self.df) - self.episode_length)
            self.current_idx = (self.episode_id * self.episode_length) % max_start

        self.episode_id += 1
        obs_vec = self._get_obs_vector(self.current_idx)
        initial_info = {
            "episode_id": self.episode_id,
            "start_index": self.current_idx,
            "ticket_id": str(self.df.iloc[self.current_idx]["ticket_id"]),
        }
        return obs_vec, initial_info

    def step(self, action_idx: int) -> StepResult:
        if action_idx < 0 or action_idx >= len(self.ACTION_LABELS):
            raise ValueError(f"action_idx {action_idx} out of range [0, {len(self.ACTION_LABELS)-1}]")

        action_label = self.ACTION_LABELS[action_idx]
        row = self.df.iloc[self.current_idx]

        true_label = str(row["true_label"])
        is_true_anomaly = bool(row["true_anomaly"])
        is_vip = bool(row["sender_is_vip"])

        obs = {
            "sender_is_vip": is_vip,
            "sender_is_internal": bool(row["sender_is_internal"]),
            "sender_tier": str(row["sender_tier"]),
            "word_count": int(row["word_count"]),
            "urgency_keyword_count": int(row["urgency_keyword_count"]),
            "angry_tone_score": float(row["angry_tone_score"]),
            "has_attachment": bool(row["has_attachment"]),
            "queue_load": int(row["queue_load"]),
        }

        truth = {
            "has_label": True,
            "label": true_label,
            "true_anomaly": is_true_anomaly,
            "is_vip": is_vip,
        }

        action = {
            "action_idx": int(action_idx),
            "label": action_label,
        }

        self.processed_count += 1
        current_queue = max(0, int(row["queue_load"]) - (1 if action_label != "spam" else 0))

        state = {
            "queue_load": current_queue,
            "processed_count": int(self.processed_count),
        }

        events = {
            "obs": obs,
            "action": action,
            "state": state,
            "truth": truth,
        }

        # Step progression
        self.step_in_episode += 1
        self.current_idx = (self.current_idx + 1) % len(self.df)

        terminated = self.step_in_episode >= self.episode_length
        truncated = False

        next_obs_vec = self._get_obs_vector(self.current_idx)

        info = {
            "ticket_id": str(row["ticket_id"]),
            "subject": str(row["subject"]),
            "body_preview": str(row["body"])[:100] + ("..." if len(str(row["body"])) > 100 else ""),
            "sender_tier": str(row["sender_tier"]),
            "sender_is_vip": is_vip,
            "angry_tone_score": float(row["angry_tone_score"]),
            "urgency_keyword_count": int(row["urgency_keyword_count"]),
            "has_attachment": bool(row["has_attachment"]),
            "true_label": true_label,
            "action_label": action_label,
            "is_correct": (action_label == true_label),
        }

        return StepResult(
            obs_vector=next_obs_vec,
            events=events,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def render_frame(
        self,
        episode_id: int,
        step_result: StepResult,
        total_reward: float,
        breakdown: dict[str, float],
    ) -> dict[str, Any]:
        """Generates structured UI telemetry payload for the ticket triage card."""
        info = step_result.info
        events = step_result.events
        return {
            "episode_id": episode_id,
            "step_in_episode": self.step_in_episode,
            "total_reward": total_reward,
            "breakdown": breakdown,
            "ticket": {
                "ticket_id": info.get("ticket_id"),
                "subject": info.get("subject"),
                "body_preview": info.get("body_preview"),
                "sender_tier": info.get("sender_tier"),
                "sender_is_vip": info.get("sender_is_vip"),
                "angry_tone_score": info.get("angry_tone_score"),
                "urgency_keyword_count": info.get("urgency_keyword_count"),
                "has_attachment": info.get("has_attachment"),
                "true_label": info.get("true_label"),
                "action_label": info.get("action_label"),
                "is_correct": info.get("is_correct"),
            },
            "events": events,
        }
