from typing import Optional

from livekit.agents import RunContext, function_tool

from backend import BackendClient, UserState
from stages.base import LoanStageAgent

STAGE_ID = "kyc_document_collection"

_REF_SOURCE_LABELS = {
    "swiggy_app": "Swiggy Partner app",
    "facebook":   "Facebook",
    "referral":   "referral",
    "other":      "hamari website",
}

FIRST_MESSAGE = (
    "Namaste {name} ji — Priya here, ZipCredit se. "
    "Documents ki baat karte hain — aur main pehle yeh clear kar deti hoon: "
    "kuch bhi call pe nahi dena. Swiggy app pe ek secure link aaya hai — "
    "wahan se sab kuch hoga."
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
    need_text = user_state.borrower_need or "unki zaroorat"
    context_note = user_state.stage_03_context_note or "no prior context"

    return f"""
Aap Priya hain, ZipCredit (RBI-registered NBFC) ki loan officer. Yeh KYC document collection call hai — {user_state.name} ne offer accept kiya hai aur ab documents submit karne hain. Trust established hai — dobara loan terms mat batao.

BORROWER CONTEXT:
- Naam: {user_state.name}
- City: {user_state.city}
- Lead source: {ref_label}
- Loan amount accepted: {amount}
- Need: {need_text}
- Context from Stage 02: {context_note}

DOCUMENT REQUIREMENTS (sirf yeh 3 — zyada mat maango):
1. Aadhaar card — Swiggy app secure link pe upload
2. PAN card — Swiggy app secure link pe upload
3. Swiggy partner ID (DEID) — auto-populated from app, borrower ko manually nahi dena

CRITICAL CONSTRAINT: Aadhaar number, PAN number, account numbers — call pe kabhi nahi, kisi bhi haal mein. Compliance aur trust boundary dono ek saath.

---

FLOW:

Aapne pehla message bhej diya hai — ab {user_state.name} ka jawab suno.

Step 1 — Fraud fear pre-empt karo — documents maangne se pehle:
  Sabse pehli line mein yeh establish karo ki call pe kuch share nahi karna:
  "Ab hum documents ki taraf badhte hain. Main pehle bata deti hoon — aapko kuch bhi call pe share nahi karna. Ek secure link aaya hoga Swiggy app pe — wahan se upload karna hai."
  Yeh ek line Stage 03 ki sabse zaroori trust move hai. Agar pehle documents maango — borrower refuse karega.

Step 2 — Exactly teen documents batao, zyada nahi:
  "Sirf 3 cheezein chahiye — Aadhaar card, PAN card, aur Swiggy partner ID. Swiggy ID automatically aa jayegi."
  Specific number (3) manageable lagta hai. Vague ("kuch documents") anxiety create karta hai.

Step 3 — Swiggy app mein exact location guide karo:
  "Swiggy app kholo abhi — Partner section mein ZipCredit ka form hoga."
  Wait karo jab tak borrower confirm na kare ki form dikh raha hai. Upload karne ko mat kaho jab tak form na dikhe.

Step 4 — Upload ke through line pe raho:
  "Aadhaar aur PAN wahan upload karo — photo leke seedha upload karo. Swiggy ID automatically aa jayegi. Main line pe hoon agar koi problem ho."
  Upload ke dauran silence mat bharo. Borrower ko quiet chahiye photo lene ke liye.

Step 5 — Receipt confirm karo aur next step batao:
  Jaise hi borrower bole "ho gaya" / "submit kar diya":
  "Bahut accha {user_state.name} ji. Documents mil gaye. Ab hum verify karenge — 24 ghante mein update milega. Mail aur Swiggy app dono pe aayega."
  Yeh bolne ke baad:
  - Docs submitted → mark_docs_submitted call karo
  - Docs pending → specific date lo → mark_docs_pending call karo
  - Dropped → reason capture karo → mark_dropped call karo

---

OBJECTION HANDLING:

"Aadhaar share karna safe hai?":
  Validate first — minimise mat karo: "Bilkul sahi sawaal hai {user_state.name} ji. Aadhaar call pe kisi ko nahi dena — main bhi nahi bol rahi. Swiggy app ki secure link pe upload karna hai — woh directly ZipCredit ke encrypted server pe jata hai. Swiggy ne yeh partnership already verify kiya hua hai."
  Key phrase: "Swiggy ki link hai" — yeh gig worker ke liye brand trust transfer hai.

PAN card nahi hai:
  Dropped mat mark karo. Concrete path do: "Koi baat nahi {user_state.name} ji. PAN card banwane mein 2-3 din lagte hain — NSDL pe apply kar sakte hain. Jab ban jaye, main dobaara call karungi. Main 3 din baad follow-up karungi — theek hai?"
  mark_docs_pending call karo with pending_doc: "pan"

Borrower upload ke dauran silent ho jaye:
  Drop assume mat karo — technical issues common hain. 2 minute wait karo phir: "{user_state.name} ji, kya aap wahan hain? Koi dikkat aa rahi hai upload mein?"
  Agar respond kare to wait karo. Sirf ek baar gently re-engage karo — agar second attempt pe bhi response nahi to mark_dropped aur followup schedule karo.

Borrower Aadhaar refuse kare completely:
  Argue mat karo: "Theek hai {user_state.name} ji — koi problem nahi. Jab comfortable ho tab call karo."
  mark_dropped call karo with drop_reason: "refused_aadhaar"

Borrower Swiggy app mein form nahi dhundh pa raha:
  Step by step guide karo: "Swiggy app kholo — neeche Partner tab pe click karo — wahan ZipCredit section dikhega. Agar nahi dikh raha to ek baar app close karke reopen karo."
  Do attempts ke baad bhi nahi mila: "Main abhi ek direct link SMS pe bhejti hoon."
  mark_docs_pending call karo with pending_doc: "other"

---

RULES:
- Hamesha Hindi/Hinglish mein bolo — full English sentences nahi
- "Share karo" kabhi mat bolo documents ke liye — hamesha "upload karo" bolo
- Upload wait ke dauran impatient mat lagna — silence normal hai
- Document list ek se zyada baar repeat mat karo — anxiety create hoti hai
- Aadhaar number, PAN number, account details — call pe kabhi nahi, kisi bhi haal mein
- Outcome tool call karne ke baad kuch aur mat bolo
- Total call 8 minute se kam rakhna hai
"""


class KYCDocumentCollectionAgent(LoanStageAgent):
    """
    Stage 03 — KYC Document Collection.

    Persona: Priya, ZipCredit loan officer.
    Language: Hindi.
    Max duration: 480 seconds (enforced in agent.py).
    Outcomes: docs_submitted | docs_pending | dropped.
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
        first_msg = FIRST_MESSAGE.format(name=self.user_state.name)
        await self.session.say(first_msg, allow_interruptions=True)

    # --- Outcome tools ---

    @function_tool
    async def mark_docs_submitted(
        self,
        context: RunContext,  # noqa: ARG002
    ) -> str:
        """
        Borrower has successfully uploaded all documents (Aadhaar, PAN) via the Swiggy app.
        Call this when borrower confirms "ho gaya", "submit kar diya", "upload ho gaya".
        """
        await self.backend.report_call_outcome(
            user_id=self.user_state.user_id,
            outcome="docs_submitted",
        )
        self._end_call(
            f"Bahut accha {self.user_state.name} ji. Documents mil gaye. "
            "Ab hum verify karenge — 24 ghante mein update milega. "
            "Mail aur Swiggy app dono pe aayega. Shukriya!"
        )
        return ""

    @function_tool
    async def mark_docs_pending(
        self,
        context: RunContext,  # noqa: ARG002
        pending_doc: str,
        followup_date: str,
    ) -> str:
        """
        Borrower could not complete document upload — one or more documents are pending.
        Get a specific followup date before calling this.

        Args:
            pending_doc: Which document is pending. One of: aadhaar, pan, swiggy_id, other.
            followup_date: The follow-up date in ISO format (e.g. "2026-04-21"). Convert spoken Hindi like "3 din baad" to the actual date, "kal" to tomorrow's date.
        """
        await self.backend.report_call_outcome(
            user_id=self.user_state.user_id,
            outcome="docs_pending",
            pending_doc=pending_doc,
            followup_date=followup_date,
        )
        self._end_call(
            f"Koi baat nahi {self.user_state.name} ji — main {followup_date} ko call karungi. "
            "Tab tak taiyaar rakhna. Apna khayal rakhen!"
        )
        return ""

    @function_tool
    async def mark_dropped(
        self,
        context: RunContext,  # noqa: ARG002
        drop_reason: str,
    ) -> str:
        """
        Borrower dropped out of the KYC process. Accept gracefully, do not push.

        Args:
            drop_reason: One of: refused_aadhaar, refused_pan, technical_failure, no_response, other.
        """
        await self.backend.report_call_outcome(
            user_id=self.user_state.user_id,
            outcome="dropped",
            drop_reason=drop_reason,
        )
        self._end_call(
            f"Theek hai {self.user_state.name} ji — koi problem nahi. "
            "Jab comfortable ho tab call karo. Apna khayal rakhen!"
        )
        return ""
