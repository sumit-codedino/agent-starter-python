from livekit.agents import RunContext, function_tool

from backend import BackendClient, UserState
from stages.base import LoanStageAgent

STAGE_ID = "offer_presentation"

_REF_SOURCE_LABELS = {
    "swiggy_app": "Swiggy Partner app",
    "facebook":   "Facebook",
    "referral":   "referral",
    "other":      "hamari website",
}

FIRST_MESSAGE = (
    "Namaste {name} ji — Priya here, ZipCredit se. "
    "Jaise baat hua tha, loan ki details leke aayi hoon. "
    "{need_line}"
    "Seedha kaam ki baat karte hain."
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
    ref_label = _REF_SOURCE_LABELS.get(user_state.ref_source, "ek platform")
    amount = _indian_amount(user_state.loan_amount_interest)
    t = user_state.loan_terms

    roi = t.get("roi_annual_pct") if "roi_annual_pct" in t else "N/A"
    tenure = t.get("tenure_months") if "tenure_months" in t else "N/A"
    emi = _indian_amount(t["emi_amount"]) if "emi_amount" in t else "N/A"
    total_repayable = _indian_amount(t["total_repayable"]) if "total_repayable" in t else "N/A"
    processing_fee = _indian_amount(t["processing_fee"]) if "processing_fee" in t else "N/A"
    net_disbursement = _indian_amount(t["net_disbursement"]) if "net_disbursement" in t else "N/A"

    need_text = user_state.borrower_need or "unki zaroorat"
    mood_text = user_state.borrower_mood or "neutral"

    return f"""
Aap Priya hain, ZipCredit (RBI-registered NBFC) ki loan officer. Yeh follow-up call hai — {user_state.name} se pehle baat ho chuki hai aur unhone loan details sunne ke liye agree kiya tha. Trust already established hai — dobara introduction ya legitimacy pitch mat karo.

BORROWER CONTEXT:
- Naam: {user_state.name}
- City: {user_state.city}
- Lead source: {ref_label}
- Loan amount: {amount}
- Need: {need_text}
- Mood from last call: {mood_text}

LOAN TERMS (yeh exact numbers use karo — approximate mat karo):
- Loan amount: {amount}
- Rate of interest: {roi}% per annum
- Tenure: {tenure} mahine — EMI: {emi} per month
- Total repayment: {total_repayable}
- Processing fee: {processing_fee} plus GST
- Net disbursement: {net_disbursement}
- Insurance: mandatory nahi hai
- Documentation: Aadhaar, PAN, Swiggy partner ID — pura digital
- Disbursement: 4 se 5 working days mein account mein

---

FLOW:

Aapne pehla message bhej diya hai — ab {user_state.name} ka jawab suno.

Step 1 — Sabse pehle paanch terms ek saath batao:
  1. Loan amount: {amount}
  2. Rate of interest: {roi}% per annum
  3. Tenure aur EMI: {tenure} mahine — {emi} per month
  4. Insurance: mandatory nahi hai
  5. Documentation: Aadhaar, PAN, Swiggy partner ID — pura digital
  Phir bolo: "4 se 5 working days mein account mein."

Step 2 — Total repayable batao:
  "Total repayment {total_repayable} hoga — koi hidden charges nahi hain."

Step 3 — Processing fee disclose karo:
  "Processing fee {processing_fee} plus GST hai — toh disbursement {net_disbursement} milega."
  "Yeh sab mail pe confirm aayega."

Step 4 — Questions ke liye floor open karo:
  "Koi sawaal hai offer ke baare mein?"
  Jawab do — agar kuch nahi pata toh bolo: "Main confirm karke mail ya call mein bata deti hoon."

Step 5 — Close karo:
  "Kya aage badhein — documentation ke liye?"
  - Agar haan → mark_offer_accepted call karo
  - Agar "sochna hai" → specific date lo → mark_callback call karo
  - Agar clear decline → reason capture karo → mark_rejected call karo

---

OBJECTION HANDLING:

"Rate zyada hai":
  "Haan, {roi}% annual hai — monthly EMI sirf {emi} padti hai. Aapke {need_text} ke liye manage ho jayega?"
  Agar phir bhi objection → callback path offer karo. Rate negotiate mat karo.

"Itna kam kyun?":
  "Abhi aapki profile ke hisaab se {amount} approve ho sakta hai. Agar yeh successfully close karte hain, next time higher amount milega."

"Total kitna bharna padega?":
  "Total {total_repayable} — {emi} har mahine, {tenure} mahine tak."

"Mujhe laga free hoga":
  "Haan, processing fee lagti hai — lekin yeh ek baar ki hai. Sab kuch transparent hai — mail pe bhi aayega."

Existing loan mention:
  "Aapka existing EMI obligation underwriting mein dekha jayega — abhi ke liye hum aage badh sakte hain."

---

RULES:
- Hamesha Hindi mein bolo
- EMI, rate of interest, CIBIL, disbursement, processing fee — yeh English terms use karo, Hindi equivalents nahi
- "Bahut accha offer hai" ya urgency language kabhi mat bolo
- Agar kuch nahi pata toh guess mat karo — "confirm karke batati hoon" bolo
- Decline ke baad push mat karo
- Outcome tool call karne ke baad kuch aur mat bolo
- Total call 5 minute se kam rakhna hai
"""


class OfferPresentationAgent(LoanStageAgent):
    """
    Stage 02 — Offer Presentation.

    Persona: Priya, ZipCredit loan officer (follow-up call).
    Language: Hindi.
    Max duration: 300 seconds (enforced in agent.py).
    Outcomes: offer_accepted | callback | rejected.
    """

    def __init__(self, user_state: UserState, backend: BackendClient) -> None:
        super().__init__(
            instructions=_build_instructions(user_state),
            user_state=user_state,
            backend=backend,
            stage_id=STAGE_ID,
        )

    async def on_enter(self) -> None:
        need_line = ""
        if self.user_state.borrower_need:
            need_line = f"{self.user_state.borrower_need} ke liye — "
        first_msg = FIRST_MESSAGE.format(
            name=self.user_state.name,
            need_line=need_line,
        )
        await self.session.say(first_msg, allow_interruptions=True)

    # --- Outcome tools ---

    @function_tool
    async def mark_offer_accepted(
        self,
        context: RunContext,  # noqa: ARG002
    ) -> str:
        """
        Borrower agreed to proceed to documentation/KYC stage.
        Call this when borrower says "haan", "aage badho", "kar do", "theek hai process karo".
        """
        await self.backend.report_call_outcome(
            user_id=self.user_state.user_id,
            outcome="offer_accepted",
            borrower_need=self.user_state.borrower_need,
        )
        self._end_call(f"Bahut achha {self.user_state.name} ji! Main aapko documentation ke liye call karungi — sab kuch digital hoga. Shukriya!")
        return ""

    @function_tool
    async def mark_callback(
        self,
        context: RunContext,  # noqa: ARG002
        callback_time: str,
    ) -> str:
        """
        Borrower wants time to think. Get a specific date/time before calling this.
        Never accept "baad mein" without a date.

        Args:
            callback_time: Specific date/time borrower agreed to (e.g. "kal shaam 5 baje", "Thursday").
        """
        await self.backend.report_call_outcome(
            user_id=self.user_state.user_id,
            outcome="callback",
            callback_time=callback_time,
        )
        self._end_call(f"Bilkul {self.user_state.name} ji — koi pressure nahi. Main {callback_time} ko call karungi. Apna khayal rakhen!")
        return ""

    @function_tool
    async def mark_rejected(
        self,
        context: RunContext,  # noqa: ARG002
        rejection_reason: str,
        any_objection_raised: bool = False,
        objection_detail: str = "",
    ) -> str:
        """
        Borrower clearly declined the offer. Accept gracefully, do not push.
        Capture the reason for rejection.

        Args:
            rejection_reason: One of: rate_too_high, amount_insufficient, changed_mind, existing_loan_concern, other
            any_objection_raised: Whether the borrower raised any objection during the call.
            objection_detail: Free text describing the objection if any.
        """
        await self.backend.report_call_outcome(
            user_id=self.user_state.user_id,
            outcome="rejected",
            rejection_reason=rejection_reason,
            any_objection_raised=any_objection_raised,
            objection_detail=objection_detail,
        )
        self._end_call(f"Theek hai {self.user_state.name} ji — koi baat nahi. Agar kabhi zaroorat ho toh mera number save kar lijiye. Apna khayal rakhen!")
        return ""
