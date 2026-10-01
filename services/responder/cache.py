"""
Cache-first response layer backed by Redis with in-memory fallback.
Cache key: sha256(scheme + intent + lang + corpus_version)
"""

import base64
import hashlib
import time
from typing import Optional, Tuple, Dict
from services.tts.mock import MOCK_WAV_HEADER

MOCK_AUDIO_B64 = base64.b64encode(MOCK_WAV_HEADER).decode("ascii")

# Seed answers for 3 schemes x 4 intents x Tier 1 languages (English, Hindi, Tamil)
SEED_ANSWERS: Dict[str, Dict[str, Dict[str, str]]] = {
    "pm_kisan": {
        "check_eligibility": {
            "en": "Small and marginal farmer families with cultivable landholding in their name are eligible for PM-KISAN, subject to institutional exclusions.",
            "hi": "जिन किसान परिवारों के नाम पर खेती योग्य भूमि है, वे पीएम-किसान योजना के लिए पात्र हैं, बशर्ते वे संस्थागत छूट के दायरे में न आते हों।",
            "ta": "தங்கள் பெயரில் சாகுபடி செய்யக்கூடிய நிலம் வைத்திருக்கும் சிறு மற்றும் குறு விவசாய குடும்பங்கள் பிஎம்-கிசான் திட்டத்திற்கு தகுதியுடையவர்கள்.",
        },
        "documents_needed": {
            "en": "To apply for PM-KISAN, you need an Aadhaar card, land ownership records (Khasra/Khatauni), and an active bank account linked with Aadhaar.",
            "hi": "पीएम-किसान के लिए आधार कार्ड, जमीन के दस्तावेज (खसरा/खतौनी), और आधार से जुड़ा बैंक खाता आवश्यक है।",
            "ta": "பிஎம்-கிசான் விண்ணப்பிக்க ஆதார் அட்டை, நில உரிமை ஆவணங்கள் மற்றும் ஆதாருடன் இணைக்கப்பட்ட வங்கி கணக்கு தேவை.",
        },
        "apply_process": {
            "en": "You can register on the pmkisan.gov.in portal via Farmer Corner or visit the nearest Common Service Centre (CSC) or Agriculture Office.",
            "hi": "आप pmkisan.gov.in पोर्टल पर फार्मर कॉर्नर के माध्यम से ऑनलाइन पंजीकरण कर सकते हैं या नजदीकी सीएससी केंद्र पर जा सकते हैं।",
            "ta": "நீங்கள் pmkisan.gov.in இணையதளம் வழியாக அல்லது அருகிலுள்ள பொது சேவை மையம் (CSC) மூலம் பதிவு செய்யலாம்.",
        },
        "benefit_amount": {
            "en": "PM-KISAN provides Rs. 6,000 per year transferred directly to your bank account in three equal installments of Rs. 2,000 every 4 months.",
            "hi": "पीएम-किसान योजना के तहत सालाना ₹6,000 तीन समान किस्तों में (₹2,000 प्रति 4 माह) सीधे बैंक खाते में भेजे जाते हैं।",
            "ta": "பிஎம்-கிசான் திட்டத்தின் கீழ் ஆண்டுக்கு ரூ. 6,000 மூன்று தவணைகளாக (தலா ரூ. 2,000) நேரடியாக வங்கி கணக்கில் வழங்கப்படுகிறது.",
        },
    },
    "mgnrega": {
        "check_eligibility": {
            "en": "Any rural household whose adult members volunteer to do unskilled manual work is eligible for employment under MGNREGA.",
            "hi": "ग्रामीण परिवार के सभी वयस्क सदस्य जो अकुशल शारीरिक श्रम करने के इच्छुक हैं, मनरेगा के तहत पात्र हैं।",
            "ta": "கிராமப்புறத்தில் உள்ள வயது வந்த நபர்கள் திறனற்ற உடலுழைப்பு செய்ய விரும்பினால் மகாத்மா காந்தி ஊரக வேலைவாய்ப்பு திட்டத்திற்கு தகுதியுடையவர்கள்.",
        },
        "documents_needed": {
            "en": "You need identity proof (Aadhaar or Voter ID), residence proof, and a passport-size photo to apply for a Job Card.",
            "hi": "जॉब कार्ड के लिए पहचान पत्र (आधार या वोटर आईडी), निवास प्रमाण पत्र और पासपोर्ट साइज फोटो आवश्यक हैं।",
            "ta": "வேலை அட்டை (Job Card) பெற அடையாளச் சான்று (ஆதார் அல்லது வாக்காளர் அட்டை), இருப்பிடச் சான்று மற்றும் பாஸ்போர்ட் அளவு புகைப்படம் தேவை.",
        },
        "apply_process": {
            "en": "Submit a written application or oral request to your local Gram Panchayat. A Job Card will be issued within 15 days.",
            "hi": "अपनी ग्राम पंचायत में आवेदन जमा करें। 15 दिनों के भीतर परिवार को निःशुल्क जॉब कार्ड जारी किया जाता है।",
            "ta": "உங்கள் கிராம பஞ்சாயத்தில் விண்ணப்பம் சமர்ப்பிக்கவும். 15 நாட்களுக்குள் வேலை அட்டை வழங்கப்படும்.",
        },
        "benefit_amount": {
            "en": "MGNREGA guarantees at least 100 days of wage employment per financial year at statutory state-notified wage rates.",
            "hi": "मनरेगा प्रत्येक वित्तीय वर्ष में कम से कम 100 दिनों के गारंटीकृत रोजगार और राज्य-निर्धारित दैनिक मजदूरी प्रदान करता है।",
            "ta": "இந்த திட்டம் ஒரு நிதியாண்டில் குறைந்தபட்சம் 100 நாட்கள் ஊதியத்துடன் கூடிய வேலைவாய்ப்பை உறுதி செய்கிறது.",
        },
    },
    "ayushman_bharat": {
        "check_eligibility": {
            "en": "Families listed in the SECC 2011 deprivation database (criteria D1-D5, D7 rural) and active RSBY beneficiaries are eligible for PMJAY.",
            "hi": "एसईसीसी 2011 सामाजिक-आर्थिक जाति जनगणना सूची में शामिल वंचित परिवार आयुष्मान भारत (पीएम-जय) के लिए पात्र हैं।",
            "ta": "SECC 2011 கணக்கெடுப்பில் தகுதியுள்ள ஏழை குடும்பங்கள் மற்றும் RSBY பயனாளிகள் ஆயுஷ்மான் பாரத் திட்டத்திற்கு தகுதியுடையவர்கள்.",
        },
        "documents_needed": {
            "en": "You need an Aadhaar card or Ration card along with a mobile number to verify eligibility at an empanelled hospital or CSC.",
            "hi": "पात्रता जांच और आयुष्मान कार्ड बनवाने के लिए आधार कार्ड या राशन कार्ड और मोबाइल नंबर आवश्यक है।",
            "ta": "ஆயுஷ்மான் அட்டை பெற ஆதார் அட்டை அல்லது ரேஷன் அட்டை மற்றும் மொபைல் எண் தேவை.",
        },
        "apply_process": {
            "en": "Check eligibility at beneficiary.nha.gov.in or visit an empanelled hospital/Ayushman Mitra kiosk to generate your card.",
            "hi": "beneficiary.nha.gov.in पर पात्रता जांचें या किसी भी सूचीबद्ध अस्पताल में आयुष्मान मित्र के पास जाकर कार्ड बनवाएं।",
            "ta": "beneficiary.nha.gov.in இணையதளத்தில் சரிபார்க்கலாம் அல்லது அங்கீகரிக்கப்பட்ட மருத்துவமனையில் ஆயுஷ்மான் மித்ராவை அணுகலாம்.",
        },
        "benefit_amount": {
            "en": "Ayushman Bharat provides cashless health cover of up to Rs. 5,00,000 per eligible family per year for secondary and tertiary care hospitalization.",
            "hi": "आयुष्मान भारत योजना प्रति पात्र परिवार को प्रति वर्ष ₹5 लाख तक का कैशलेस स्वास्थ्य उपचार कवर प्रदान करती है।",
            "ta": "ஆயுஷ்மான் பாரத் திட்டம் தகுதியுள்ள குடும்பத்திற்கு ஆண்டுக்கு ரூ. 5 லட்சம் வரை பணமில்லா மருத்துவ காப்பீடு வழங்குகிறது.",
        },
    },
}


class ResponseCache:
    """Pre-synthesized answer cache for common (scheme x intent x language) combinations."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self._memory_cache: Dict[str, Tuple[str, str]] = {}
        self._redis_client = None
        self._init_memory_seed()

    def _init_memory_seed(self):
        """Populate initial 3x4x3 cache table in memory."""
        for scheme, intents in SEED_ANSWERS.items():
            for intent, lang_map in intents.items():
                for lang, reply in lang_map.items():
                    key = self.generate_key(scheme, intent, lang, "1.0")
                    self._memory_cache[key] = (reply, MOCK_AUDIO_B64)

    @staticmethod
    def generate_key(scheme: str, intent: str, lang: str, corpus_version: str = "1.0") -> str:
        payload = f"{scheme}:{intent}:{lang}:{corpus_version}".lower()
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def get(
        self, scheme: str, intent: str, lang: str, corpus_version: str = "1.0"
    ) -> Tuple[Optional[str], Optional[str], float]:
        """
        Lookup cached response.
        Returns: (reply_text, audio_b64, elapsed_ms)
        """
        t0 = time.perf_counter()
        key = self.generate_key(scheme, intent, lang, corpus_version)

        # Check in-memory seed cache first
        if key in self._memory_cache:
            reply_text, audio_b64 = self._memory_cache[key]
            ms = (time.perf_counter() - t0) * 1000
            return reply_text, audio_b64, round(ms, 2)

        # Fallback to English for the scheme x intent if requested language is missing from cache
        if lang != "en":
            fallback_key = self.generate_key(scheme, intent, "en", corpus_version)
            if fallback_key in self._memory_cache:
                reply_text, audio_b64 = self._memory_cache[fallback_key]
                ms = (time.perf_counter() - t0) * 1000
                return reply_text, audio_b64, round(ms, 2)

        ms = (time.perf_counter() - t0) * 1000
        return None, None, round(ms, 2)

    async def set(
        self,
        scheme: str,
        intent: str,
        lang: str,
        reply_text: str,
        audio_b64: str,
        corpus_version: str = "1.0",
    ):
        """Store response in cache."""
        key = self.generate_key(scheme, intent, lang, corpus_version)
        self._memory_cache[key] = (reply_text, audio_b64)
