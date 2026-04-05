from livekit.agents import RunContext, function_tool

from backend import BackendClient, UserState
from stages.base import LoanStageAgent

STAGE_ID = "cold_call"

_REF_SOURCE_LABELS = {
    "swiggy_app": "Swiggy Partner app",
    "facebook":   "Facebook",
    "referral":   "referral",
    "other":      "hamari website",
}

# First message is fixed per the conversation design
FIRST_MESSAGE = "Hi {name}, main Priya bol rahi hoon ZipCredit se. Kya aapke paas 30 second hain?"


def _indian_amount(n: int) -> str:
    """Convert a number to spoken Indian format, e.g. 150000 → '1 lakh 50 hazaar'."""
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

    return f"""
Aap Priya hain, ZipCredit (RBI-registered NBFC) ki ek confident loan officer. Aap {user_state.name} ko call kar rahi hain jo {user_state.city} mein rehte hain aur {ref_label} ke zariye {amount} ke loan mein interest dikhaya tha.

LAKSHYA: Sirf interest qualify karna hai. EMI, rate, ya documents kabhi mat batao — ye Stage 2 ka kaam hai.

---

Aapne apna pehla message bhej diya hai:
"Hi {user_state.name}, main Priya bol rahi hoon ZipCredit se. Kya aapke paas 30 second hain?"

Ab {user_state.name} ka jawab suno aur neeche diye branch follow karo:

--- BRANCH A: Borrower available hai (haan / theek hai / boliye / sure) ---

  Step 1 — Lead reference do:
    "Aapne haal hi mein {ref_label} pe {amount} ke loan mein interest dikhaya tha — main usi ke baare mein call kar rahi thi."

  Step 2 — Agar suspicious lage sirf tab:
    "Bilkul samajh sakti hoon. Aapko {ref_label} pe ek confirmation message aaya hoga."
    Phir ruko — zyada explain mat karo.

  Step 3 — Need poochho:
    "{amount} ka loan kisi khaas cheez ke liye tha?"
    Jawab suno aur note karo.

  Step 4 — Follow-up time lo:
    "Theek hai! Main details ke saath call karungi — aaj kaunsa time achha rahega?"
    Specific time lo (e.g. "shaam 4 baje"). Vague jawab ("kal", "baad mein") accept mat karo.

  Step 5 — mark_hot_lead call karo with borrower_need + follow_up_time.

--- BRANCH B: Borrower busy hai (nahi / abhi nahi / baad mein) ---

  Step 1 — Specific time lo:
    "Koi baat nahi — kab call karoon? Ek specific time batao."
    Exact time lo. "Baad mein" accept mat karo.

  Step 2 — mark_callback call karo with callback_time.

--- BRANCH C: Clear decline (nahi chahiye / interest nahi) ---

  mark_not_interested call karo. Push mat karo, gracefully end karo.

--- BRANCH D: Koi response nahi / call cut gaya ---

  mark_no_response call karo.

---

RULES:
- Hamesha Hindi mein bolo
- Warm, calm, confident — hesitation se trust kam hota hai
- Short aur natural responses — scripted mat lagna chahiye
- EMI, interest rate, documents, penalties kabhi mention mat karo
- Ek baar branch decide ho jaye toh dobara permission mat maango
- Outcome tool call karne ke baad kuch aur mat bolo
- Total call 3 minute se kam rakhna hai
"""


class ColdCallAgent(LoanStageAgent):
    """
    Stage 01 — Cold call / qualify.

    Persona: Priya, ZipCredit loan officer.
    Language: Hindi.
    Max duration: 180 seconds (enforced in agent.py).
    Outcomes: hot_lead | callback | not_interested | no_response.
    """

    def __init__(self, user_state: UserState, backend: BackendClient) -> None:
        super().__init__(
            instructions=_build_instructions(user_state),
            user_state=user_state,
            backend=backend,
            stage_id=STAGE_ID,
        )

    async def on_enter(self) -> None:
        first_msg = FIRST_MESSAGE.format(name=self.user_state.name)
        await self.session.say(first_msg, allow_interruptions=True)

    # --- Outcome tools ---

    @function_tool
    async def mark_hot_lead(
        self,
        context: RunContext,  # noqa: ARG002
        borrower_need: str,
        follow_up_time: str,
    ) -> str:
        """
        Borrower is interested and has confirmed a follow-up time. This is a hot lead.
        Only call this after you have both borrower_need AND a specific follow_up_time.

        Args:
            borrower_need: What the borrower said the money is for (e.g. "bike repair", "medical").
            follow_up_time: The specific time confirmed for the next call (e.g. "aaj shaam 4 baje").
        """
        await self.backend.report_call_outcome(
            user_id=self.user_state.user_id,
            outcome="hot_lead",
            borrower_need=borrower_need,
        )
        self._end_call(f"Bahut achha {self.user_state.name}! Main aapko {follow_up_time} pe call karungi details ke saath. Tab tak apna khayal rakhen!")
        return ""

    @function_tool
    async def mark_callback(
        self,
        context: RunContext,  # noqa: ARG002
        callback_time: str,
        borrower_need: str = "",
    ) -> str:
        """
        Borrower is busy now but gave a specific time to call back.
        Only call this after getting an exact time — never accept vague answers.

        Args:
            callback_time: Specific callback time given by borrower (e.g. "aaj shaam 6 baje").
            borrower_need: What the borrower mentioned needing money for, if shared.
        """
        await self.backend.report_call_outcome(
            user_id=self.user_state.user_id,
            outcome="callback",
            callback_time=callback_time,
            borrower_need=borrower_need or None,
        )
        self._end_call(f"Bilkul {self.user_state.name} — main aapko {callback_time} pe call karungi. Shukriya!")
        return ""

    @function_tool
    async def mark_not_interested(self, context: RunContext) -> str:  # noqa: ARG002
        """
        Borrower clearly declined. Accept gracefully and end immediately.
        Do not push after a clear no.
        """
        await self.backend.report_call_outcome(
            user_id=self.user_state.user_id,
            outcome="not_interested",
        )
        self._end_call(f"Theek hai {self.user_state.name} — koi baat nahi. Apna khayal rakhen!")
        return ""

    @function_tool
    async def mark_no_response(self, context: RunContext) -> str:  # noqa: ARG002
        """
        Borrower hung up, did not engage, or line dropped.
        Use when there is no meaningful conversation.
        """
        await self.backend.report_call_outcome(
            user_id=self.user_state.user_id,
            outcome="no_response",
        )
        self._end_call()
        return ""
