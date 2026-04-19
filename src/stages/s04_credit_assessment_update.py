from typing import Optional

from livekit.agents import RunContext, function_tool

from backend import BackendClient, UserState
from stages.base import LoanStageAgent

STAGE_ID = "credit_assessment_update"

FIRST_MESSAGE_APPROVED = (
    "Namaste {name} ji — Priya here, ZipCredit se. "
    "Acchi khabar hai — aapka loan approve ho gaya hai! "
    "{approved_amount}, {roi}% rate, {tenure} mahine."
)

FIRST_MESSAGE_REJECTED = (
    "Namaste {name} ji — Priya here, ZipCredit se. "
    "Assessment ka result aaya hai — main explain karti hoon."
)

FIRST_MESSAGE_MORE_INFO = (
    "Namaste {name} ji — Priya here, ZipCredit se. "
    "Ek document incomplete lag raha hai — {missing_doc}. "
    "Main explain karti hoon."
)


def _indian_amount(n: int) -> str:
    lakhs = n // 100_000
    thousands = (n % 100_000) // 1_000
    remainder = n % 1_000
    parts = []
    if lakhs:
        parts.append(f"{lakhs} lakh")
    if thousands:
        parts.append(f"{thousands} hazaar")
    if remainder:
        parts.append(str(remainder))
    return " ".join(parts) if parts else "0"


def _build_instructions(user_state: UserState) -> str:
    amount = _indian_amount(user_state.loan_amount_interest)
    approved_amount = _indian_amount(user_state.approved_amount) if user_state.approved_amount else "N/A"
    assessment = user_state.assessment_outcome or "approved"
    rejection_reason = user_state.rejection_reason or ""
    next_step = user_state.next_step_if_rejected or ""
    missing_doc = user_state.missing_doc or ""
    doc_issue = user_state.doc_issue or ""

    t = user_state.loan_terms
    roi = t.get("roi_annual_pct", "N/A") if t else "N/A"
    tenure = t.get("tenure_months", "N/A") if t else "N/A"

    return f"""
Aap Priya hain, ZipCredit (RBI-registered NBFC) ki loan officer. Yeh credit assessment result delivery call hai. Trust established hai — dobara introduction mat karo.

BORROWER CONTEXT:
- Naam: {user_state.name}
- City: {user_state.city}
- Assessment outcome: {assessment}
- Approved amount: {approved_amount}
- Rejection reason: {rejection_reason}
- Next step if rejected: {next_step}

DECISION CONTEXT:
- ROI: {roi}% per annum
- Tenure: {tenure} mahine
- Missing doc (if more info needed): {missing_doc}
- Doc issue detail: {doc_issue}

---

FLOW:

Yeh ek news delivery call hai — borrower ka jawab suno aur react karo.

Step 1 — Pehle sentence mein outcome batao — suspense mat banao:

  Approved:
    "Acchi khabar hai — aapka loan approve ho gaya hai! {approved_amount}, {roi}% rate, {tenure} mahine."

  Rejected:
    "Assessment mein ek issue aaya hai — {rejection_reason}. Main explain karti hoon."

  More info needed:
    "Ek document incomplete lag raha hai — {missing_doc}. Aap kuch din mein upload kar do, process continue ho jayega."

Step 2 (approved) — Amount confirm karo aur next step batao:
  "Ab hum agreement aur consent pe aayenge — ek aur short call hogi aaj ya kal. Main schedule karungi."
  Brief raho — borrower khush hai, over-explain mat karo.
  Stage 05 slot lock karo before ending → mark_acknowledged_approved call karo

Step 3 (rejected) — Reason plain Hinglish mein batao, turant next step do:
  Rejection bina next step ke = permanent borrower loss.
  - Low CIBIL: "Abhi CIBIL score thoda low hai — kuch mahine mein improve ho sakta hai. Main ek SMS bhejti hoon tips ke saath. Tab reapply kar sakte hain."
  - Income insufficient: "Abhi income threshold se thoda kam hai. Agar salary badhti hai ya co-applicant add karte hain, reapply kar sakte hain."
  - Document error: "Ek document mein issue tha — {doc_issue}. Correct document re-upload karo, main 48 ghante mein re-assess karungi."
  → mark_acknowledged_rejected call karo

Step 4 (more info needed) — Exactly kya missing hai aur kab tak chahiye:
  "Bas {missing_doc} chahiye. Swiggy app ki secure link pe kuch din mein upload kar do — process continue ho jayega. Main follow-up karungi."
  → mark_query_raised call karo

---

OBJECTION HANDLING:

Borrower upset about rejection:
  Validate first: "Samajh mein aata hai {user_state.name} ji — disappointing hota hai. Lekin yeh permanent nahi hai." Phir next step do. Decision ke liye apologise mat karo — yeh aapne nahi kiya.

Borrower appeal maangta hai:
  "Abhi manual review nahi hota — lekin kuch mahine mein reapply karne pe fresh assessment hoga." Override promise mat karo.

Approved amount kam hai:
  "Abhi aapki profile ke hisaab se {approved_amount} approve hua. Yeh aapka first loan hai — next time amount improve ho sakta hai."

Reapply karne ka timeframe poochha:
  "Generally 3-6 mahine baad — jab CIBIL improve ho jaye. Main SMS pe guide kar deti hoon."

Borrower rejection ke baad silent ya emotionally flat:
  3 second wait karo. Phir: "Koi sawaal hai ya main aapko SMS pe sab kuch bhej dun?" Easy exit do.

---

RULES:
- Approved news warmth se batao — lekin "bahut bahut badhaai" repeat mat karo
- Rejected news empathy se batao — "mujhe bahut dukh hai" kabhi mat bolo
- Hamesha next step do — rejection ko dead end mat chhoddo
- Passive language nahi — "rejection ho gayi" nahi, "abhi approve nahi hua, lekin {{next step}}" bolo
- Decision ke baad turant call close karo — over-explain mat karo
- Outcome tool call karne ke baad kuch aur mat bolo
- Total call 4 minute se kam rakhna hai
"""


class CreditAssessmentUpdateAgent(LoanStageAgent):
    """
    Stage 04 — Credit Assessment Update.

    Persona: Priya, ZipCredit loan officer.
    Language: Hindi.
    Max duration: 240 seconds (enforced in agent.py).
    Outcomes: acknowledged_approved | acknowledged_rejected | query_raised.
    """

    def __init__(
        self,
        user_state: UserState,
        backend: BackendClient,
        template: Optional[str] = None,  # noqa: ARG002 — disabled for now
        first_message: Optional[str] = None,  # noqa: ARG002 — disabled for now
    ) -> None:
        super().__init__(
            instructions=_build_instructions(user_state),
            user_state=user_state,
            backend=backend,
            stage_id=STAGE_ID,
        )

    async def on_enter(self) -> None:
        assessment = self.user_state.assessment_outcome or "approved"
        t = self.user_state.loan_terms or {}
        roi = t.get("roi_annual_pct", "N/A")
        tenure = t.get("tenure_months", "N/A")
        approved_amount = _indian_amount(self.user_state.approved_amount) if self.user_state.approved_amount else "N/A"
        missing_doc = self.user_state.missing_doc or ""

        if assessment == "approved":
            msg = FIRST_MESSAGE_APPROVED.format(
                name=self.user_state.name,
                approved_amount=approved_amount,
                roi=roi,
                tenure=tenure,
            )
        elif assessment == "more_info_needed":
            msg = FIRST_MESSAGE_MORE_INFO.format(
                name=self.user_state.name,
                missing_doc=missing_doc,
            )
        else:
            msg = FIRST_MESSAGE_REJECTED.format(name=self.user_state.name)

        await self.session.say(msg, allow_interruptions=True)

    # --- Outcome tools ---

    @function_tool
    async def mark_acknowledged_approved(
        self,
        context: RunContext,  # noqa: ARG002
        borrower_intent: str,
    ) -> str:
        """
        Borrower acknowledged the approval and is ready for next steps.
        Call this after explaining the approved terms and the borrower confirms.

        Args:
            borrower_intent: Borrower's intent after hearing the approval. One of: proceeding, reconsidering.
        """
        await self.backend.report_call_outcome(
            user_id=self.user_state.user_id,
            outcome="acknowledged_approved",
            borrower_intent=borrower_intent,
        )
        self._end_call(
            f"Bahut achha {self.user_state.name} ji! Agreement call ke liye main schedule karungi — "
            "aaj ya kal. Shukriya!"
        )
        return ""

    @function_tool
    async def mark_acknowledged_rejected(
        self,
        context: RunContext,  # noqa: ARG002
        rejection_reason: str,
    ) -> str:
        """
        Borrower acknowledged the rejection. Always provide a next step before calling this.

        Args:
            rejection_reason: One of: low_cibil, income_insufficient, doc_error, employment_type, other.
        """
        await self.backend.report_call_outcome(
            user_id=self.user_state.user_id,
            outcome="acknowledged_rejected",
            rejection_reason=rejection_reason,
        )
        self._end_call(
            f"Theek hai {self.user_state.name} ji — SMS pe sab kuch bhej rahi hoon. "
            "Jab ready ho, reapply kar sakte hain. Apna khayal rakhen!"
        )
        return ""

    @function_tool
    async def mark_query_raised(
        self,
        context: RunContext,  # noqa: ARG002
        pending_doc: str,
        followup_date: str,
    ) -> str:
        """
        More information is needed — borrower needs to upload a missing/corrected document.
        Get a specific followup date before calling this.

        Args:
            pending_doc: Which document is missing or needs correction (e.g. "aadhaar", "pan", "income_proof").
            followup_date: The follow-up date in ISO format (e.g. "2026-04-22"). Convert spoken Hindi like "2 din baad" to the actual date.
        """
        await self.backend.report_call_outcome(
            user_id=self.user_state.user_id,
            outcome="query_raised",
            pending_doc=pending_doc,
            followup_date=followup_date,
        )
        self._end_call(
            f"Theek hai {self.user_state.name} ji — {followup_date} ko main follow-up karungi. "
            "Swiggy app pe upload kar dena. Shukriya!"
        )
        return ""
