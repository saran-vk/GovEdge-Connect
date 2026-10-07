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
            "te": "తమ పేరుపై సాగు చేయదగిన భూమి ఉన్న చిన్న మరియు సన్నకారు రైతు కుటుంబాలు PM-KISAN పథకానికి అర్హులు, సంస్థాగత మినహాయింపులకు లోబడి.",
            "ml": "സ്വന്തം പേരിൽ കൃഷിഭൂമിയുള്ള ചെറുകിട, നാമമാത്ര കർഷക കുടുംബങ്ങൾക്ക് പിഎം-കിസാൻ പദ്ധതിക്ക് അർഹതയുണ്ട്, സ്ഥാപനപരമായ ഒഴിവാക്കലുകൾക്ക് വിധേയമായി.",
        },
        "documents_needed": {
            "en": "To apply for PM-KISAN, you need an Aadhaar card, land ownership records (Khasra/Khatauni), and an active bank account linked with Aadhaar.",
            "hi": "पीएम-किसान के लिए आधार कार्ड, जमीन के दस्तावेज (खसरा/खतौनी), और आधार से जुड़ा बैंक खाता आवश्यक है।",
            "ta": "பிஎம்-கிசான் விண்ணப்பிக்க ஆதார் அட்டை, நில உரிமை ஆவணங்கள் மற்றும் ஆதாருடன் இணைக்கப்பட்ட வங்கி கணக்கு தேவை.",
            "te": "PM-KISAN కోసం ఆధార్ కార్డు, భూమి యాజమాన్య పత్రాలు (ఖస్రా/ఖతౌని), మరియు ఆధార్‌తో అనుసంధానించబడిన క్రియాశీల బ్యాంక్ ఖాతా అవసరం.",
            "ml": "പിഎം-കിസാന് അപേക്ഷിക്കാൻ ആധാർ കാർഡ്, ഭൂവുടമസ്ഥാവകാശ രേഖകൾ (ഖസ്ര/ഖതൗണി), ആധാറുമായി ബന്ധിപ്പിച്ച ബാങ്ക് അക്കൗണ്ട് എന്നിവ ആവശ്യമാണ്.",
        },
        "apply_process": {
            "en": "You can register on the pmkisan.gov.in portal via Farmer Corner or visit the nearest Common Service Centre (CSC) or Agriculture Office.",
            "hi": "आप pmkisan.gov.in पोर्टल पर फार्मर कॉर्नर के माध्यम से ऑनलाइन पंजीकरण कर सकते हैं या नजदीकी सीएससी केंद्र पर जा सकते हैं।",
            "ta": "நீங்கள் pmkisan.gov.in இணையதளம் வழியாக அல்லது அருகிலுள்ள பொது சேவை மையம் (CSC) மூலம் பதிவு செய்யலாம்.",
            "te": "మీరు pmkisan.gov.in పోర్టల్‌లో ఫార్మర్ కార్నర్ ద్వారా ఆన్‌లైన్‌లో నమోదు చేసుకోవచ్చు లేదా సమీపంలోని కామన్ సర్వీస్ సెంటర్ (CSC) ని సంప్రదించవచ్చు.",
            "ml": "നിങ്ങൾക്ക് pmkisan.gov.in പോർട്ടൽ വഴി ഫാർമേഴ്‌സ് കോർണറിലൂടെയോ അടുത്തുള്ള അക്ഷയ/CSC കേന്ദ്രം വഴിയോ രജിസ്റ്റർ ചെയ്യാം.",
        },
        "benefit_amount": {
            "en": "PM-KISAN provides Rs. 6,000 per year transferred directly to your bank account in three equal installments of Rs. 2,000 every 4 months.",
            "hi": "पीएम-किसान योजना के तहत सालाना ₹6,000 तीन समान किस्तों में (₹2,000 प्रति 4 माह) सीधे बैंक खाते में भेजे जाते हैं।",
            "ta": "பிஎம்-கிசான் திட்டத்தின் கீழ் ஆண்டுக்கு ரூ. 6,000 மூன்று தவணைகளாக (தலா ரூ. 2,000) நேரடியாக வங்கி கணக்கில் வழங்கப்படுகிறது.",
            "te": "PM-KISAN పథకం కింద సంవత్సరానికి రూ. 6,000 నేరుగా బ్యాంక్ ఖాతాకు ప్రతి 4 నెలలకు రూ. 2,000 చొప్పున మూడు విడతలలో జమ చేయబడుతుంది.",
            "ml": "പിഎം-കിസാൻ പദ്ധതി വഴി പ്രതിവർഷം 6,000 രൂപ 4 മാസം കൂടുമ്പോൾ 2,000 രൂപ വീതം മൂന്ന് തുല്യ ഗഡുക്കളായി ബാങ്ക് അക്കൗണ്ടിലേക്ക് നൽകുന്നു.",
        },
    },
    "mgnrega": {
        "check_eligibility": {
            "en": "Any rural household whose adult members volunteer to do unskilled manual work is eligible for employment under MGNREGA.",
            "hi": "ग्रामीण परिवार के सभी वयस्क सदस्य जो अकुशल शारीरिक श्रम करने के इच्छुक हैं, मनरेगा के तहत पात्र हैं।",
            "ta": "கிராமப்புறத்தில் உள்ள வயது வந்த நபர்கள் திறனற்ற உடலுழைப்பு செய்ய விரும்பினால் மகாத்மா காந்தி ஊரக வேலைவாய்ப்பு திட்டத்திற்கு தகுதியுடையவர்கள்.",
            "te": "నైపుణ్యం లేని శారీరక శ్రమ చేయడానికి సిద్ధంగా ఉన్న గ్రామీణ కుటుంబాల వయోజన సభ్యులు ఉపాధి హామీ పథకానికి (MGNREGA) అర్హులు.",
            "ml": "അവിദഗ്ദ്ധ കായിക അധ്വാനം ചെയ്യാൻ സന്നദ്ധരായ ഗ്രാമീണ കുടുംബങ്ങളിലെ പ്രായപൂർത്തിയായ അംഗങ്ങൾക്ക് തൊഴിലുറപ്പ് പദ്ധതിക്ക് (MGNREGA) അർഹതയുണ്ട്.",
        },
        "documents_needed": {
            "en": "You need identity proof (Aadhaar or Voter ID), residence proof, and a passport-size photo to apply for a Job Card.",
            "hi": "जॉब कार्ड के लिए पहचान पत्र (आधार या वोटर आईडी), निवास प्रमाण पत्र और पासपोर्ट साइज फोटो आवश्यक हैं।",
            "ta": "வேலை அட்டை (Job Card) பெற அடையாளச் சான்று (ஆதார் அல்லது வாக்காளர் அட்டை), இருப்பிடச் சான்று மற்றும் பாஸ்போர்ட் அளவு புகைப்படம் தேவை.",
            "te": "జాబ్ కార్డు దరఖాస్తుకు గుర్తింపు కార్డు (ఆధార్ లేదా ఓటర్ ఐడీ), నివాస ధృవీకరణ మరియు పాస్‌పోర్ట్ సైజు ఫోటో అవసరం.",
            "ml": "ജോബ് കാർഡിനായി തിരിച്ചറിയൽ രേഖ (ആധാർ അല്ലെങ്കിൽ വോട്ടർ ഐഡി), താമസരേഖ, പാസ്‌പോർട്ട് സൈസ് ഫോട്ടോ എന്നിവ ആവശ്യമാണ്.",
        },
        "apply_process": {
            "en": "Submit a written application or oral request to your local Gram Panchayat. A Job Card will be issued within 15 days.",
            "hi": "अपनी ग्राम पंचायत में आवेदन जमा करें। 15 दिनों के भीतर परिवार को निःशुल्क जॉब कार्ड जारी किया जाता है।",
            "ta": "உங்கள் கிராம பஞ்சாயத்தில் விண்ணப்பம் சமர்ப்பிக்கவும். 15 நாட்களுக்குள் வேலை அட்டை வழங்கப்படும்.",
            "te": "మీ స్థానిక గ్రామ పంచాయతీలో దరఖాస్తు సమర్పించండి. 15 రోజుల్లోగా కుటుంబానికి ఉచితంగా జాబ్ కార్డు జారీ చేయబడుతుంది.",
            "ml": "നിങ്ങളുടെ ഗ്രാമപഞ്ചായത്തിൽ അപേക്ഷ നൽകുക. 15 ദിവസത്തിനകം സൗജന്യമായി ജോബ് കാർഡ് ലഭിക്കും.",
        },
        "benefit_amount": {
            "en": "MGNREGA guarantees at least 100 days of wage employment per financial year at statutory state-notified wage rates.",
            "hi": "मनरेगा प्रत्येक वित्तीय वर्ष में कम से कम 100 दिनों के गारंटीकृत रोजगार और राज्य-निर्धारित दैनिक मजदूरी प्रदान करता है।",
            "ta": "இந்த திட்டம் ஒரு நிதியாண்டில் குறைந்தபட்சம் 100 நாட்கள் ஊதியத்துடன் கூடிய வேலைவாய்ப்பை உறுதி செய்கிறது.",
            "te": "ఉపాధి హామీ పథకం ప్రతి ఆర్థిక సంవత్సరంలో కనీసం 100 రోజుల వేతన ఉపాధిని మరియు రాష్ట్ర ప్రభుత్వం నిర్ణయించిన రోజువారీ కూలీని హామీ ఇస్తుంది.",
            "ml": "തൊഴിലുറപ്പ് പദ്ധതി ഓരോ സാമ്പത്തിക വർഷത്തിലും കുറഞ്ഞത് 100 ദിവസത്തെ വേതനത്തോടുകൂടിയ തൊഴിലും സർക്കാർ നിശ്ചയിച്ച വേതനവും ഉറപ്പുനൽകുന്നു.",
        },
    },
    "ayushman_bharat": {
        "check_eligibility": {
            "en": "Families listed in the SECC 2011 deprivation database (criteria D1-D5, D7 rural) and active RSBY beneficiaries are eligible for PMJAY.",
            "hi": "एसईसीसी 2011 सामाजिक-आर्थिक जाति जनगणना सूची में शामिल वंचित परिवार आयुष्मान भारत (पीएम-जय) के लिए पात्र हैं।",
            "ta": "SECC 2011 கணக்கெடுப்பில் தகுதியுள்ள ஏழை குடும்பங்கள் மற்றும் RSBY பயனாளிகள் ஆயுஷ்மான் பாரத் திட்டத்திற்கு தகுதியுடையவர்கள்.",
            "te": "SECC 2011 సామాజిక-ఆర్థిక కుల గణన జాబితాలో ఉన్న పేద కుటుంబాలు మరియు క్రియాశీల RSBY లబ్ధిదారులు ఆయుష్మాన్ భారత్ (PM-JAY) పథకానికి అర్హులు.",
            "ml": "SECC 2011 ലിസ്റ്റിൽ ഉൾപ്പെട്ട അർഹരായ കുടുംബങ്ങൾക്കും RSBY ഗുണഭോക്താക്കൾക്കും ആയുഷ്മാൻ ഭാരത് (PMJAY) പദ്ധതിക്ക് അർഹതയുണ്ട്.",
        },
        "documents_needed": {
            "en": "You need an Aadhaar card or Ration card along with a mobile number to verify eligibility at an empanelled hospital or CSC.",
            "hi": "पात्रता जांच और आयुष्मान कार्ड बनवाने के लिए आधार कार्ड या राशन कार्ड और मोबाइल नंबर आवश्यक है।",
            "ta": "ஆயுஷ்மான் அட்டை பெற ஆதார் அட்டை அல்லது ரேஷன் அட்டை மற்றும் மொபைல் எண் தேவை.",
            "te": "అర్హత ధృవీకరణ మరియు ఆయుష్మాన్ కార్డు పొందడానికి ఆధార్ కార్డు లేదా రేషన్ కార్డు మరియు మొబైల్ నంబర్ అవసరం.",
            "ml": "ആയുഷ്മാൻ കാർഡ് എടുക്കുന്നതിനായി ആധാർ കാർഡോ റേഷൻ കാർഡോ മൊബൈൽ നമ്പറോ ആവശ്യമാണ്.",
        },
        "apply_process": {
            "en": "Check eligibility at beneficiary.nha.gov.in or visit an empanelled hospital/Ayushman Mitra kiosk to generate your card.",
            "hi": "beneficiary.nha.gov.in पर पात्रता जांचें या किसी भी सूचीबद्ध अस्पताल में आयुष्मान मित्र के पास जाकर कार्ड बनवाएं।",
            "ta": "beneficiary.nha.gov.in இணையதளத்தில் சரிபார்க்கலாம் அல்லது அங்கீகரிக்கப்பட்ட மருத்துவமனையில் ஆயுஷ்மான் மித்ராவை அணுகலாம்.",
            "te": "beneficiary.nha.gov.in పోర్టల్‌లో అర్హతను తనిఖీ చేయండి లేదా జాబితా చేయబడిన ఆసుపత్రిలోని ఆయుష్మాన్ మిత్ర సహాయంతో కార్డు పొందండి.",
            "ml": "beneficiary.nha.gov.in വഴി യോഗ്യത പരിശോധിക്കാം അല്ലെങ്കിൽ അംഗീകൃത ആശുപത്രിയിലെ ആയുഷ്മാൻ മിത്ര വഴി കാർഡ് എടുക്കാം.",
        },
        "benefit_amount": {
            "en": "Ayushman Bharat provides cashless health cover of up to Rs. 5,00,000 per eligible family per year for secondary and tertiary care hospitalization.",
            "hi": "आयुष्मान भारत योजना प्रति पात्र परिवार को प्रति वर्ष ₹5 लाख तक का कैशलेस स्वास्थ्य उपचार कवर प्रदान करती है।",
            "ta": "ஆயுஷ்மான் பாரத் திட்டம் தகுதியுள்ள குடும்பத்திற்கு ஆண்டுக்கு ரூ. 5 லட்சம் வரை பணமில்லா மருத்துவ காப்பீடு வழங்குகிறது.",
            "te": "ఆయుష్మాన్ భారత్ పథకం ప్రతి అర్హత కలిగిన కుటుంబానికి ద్వితీయ మరియు తృతీయ స్థాయి చికిత్సలకు సంవత్సరానికి రూ. 5,00,000 వరకు నగదు రహిత ఆరోగ్య బీమాను అందిస్తుంది.",
            "ml": "ആയുഷ്മാൻ ഭാരത് പദ്ധതി അർഹരായ കുടുംബത്തിന് ദ്വിതീയ, തൃതീയ ചികിത്സകൾക്കായി പ്രതിവർഷം 5 ലക്ഷം രൂപ വരെ സൗജന്യ ചികിത്സാ പരിരക്ഷ നൽകുന്നു.",
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
            if audio_b64 == MOCK_AUDIO_B64:
                try:
                    from services.tts.speech_engine import get_cached_speech_b64
                    audio_b64 = get_cached_speech_b64(reply_text, lang)
                    self._memory_cache[key] = (reply_text, audio_b64)
                except Exception:
                    pass
            ms = (time.perf_counter() - t0) * 1000
            return reply_text, audio_b64, round(ms, 2)

        # Fallback to English for the scheme x intent if requested language is missing from cache
        if lang != "en":
            fallback_key = self.generate_key(scheme, intent, "en", corpus_version)
            if fallback_key in self._memory_cache:
                reply_text, audio_b64 = self._memory_cache[fallback_key]
                if audio_b64 == MOCK_AUDIO_B64:
                    try:
                        from services.tts.speech_engine import get_cached_speech_b64
                        audio_b64 = get_cached_speech_b64(reply_text, "en")
                        self._memory_cache[fallback_key] = (reply_text, audio_b64)
                    except Exception:
                        pass
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
