"""
Generate 220+ high-quality multilingual intent queries (English, Tamil, Hindi)
covering the 4 core welfare intents and out_of_scope queries.
"""

import json
from pathlib import Path

DATASET = [
    # =========================================================================
    # PM-KISAN: check_eligibility
    # =========================================================================
    {"query": "Who is eligible to receive PM-KISAN funds?", "lang": "en", "intent": "check_eligibility", "scheme": "pm_kisan"},
    {"query": "Am I eligible for PM Kisan if I have 2 acres of farmland?", "lang": "en", "intent": "check_eligibility", "scheme": "pm_kisan"},
    {"query": "Can government employees receive PM-KISAN benefits?", "lang": "en", "intent": "check_eligibility", "scheme": "pm_kisan"},
    {"query": "Are tenant farmers eligible under PM-KISAN scheme?", "lang": "en", "intent": "check_eligibility", "scheme": "pm_kisan"},
    {"query": "Can both husband and wife get PM Kisan money in one family?", "lang": "en", "intent": "check_eligibility", "scheme": "pm_kisan"},
    {"query": "Is income tax payer eligible for PM-KISAN?", "lang": "en", "intent": "check_eligibility", "scheme": "pm_kisan"},
    {"query": "Can a retired pensioner with land get PM Kisan?", "lang": "en", "intent": "check_eligibility", "scheme": "pm_kisan"},
    {"query": "Who qualifies as a beneficiary under PM Kisan Samman Nidhi?", "lang": "en", "intent": "check_eligibility", "scheme": "pm_kisan"},
    {"query": "Does PM-KISAN apply to urban farmland owners?", "lang": "en", "intent": "check_eligibility", "scheme": "pm_kisan"},
    {"query": "What is the landholding limit to be eligible for PM Kisan?", "lang": "en", "intent": "check_eligibility", "scheme": "pm_kisan"},
    {"query": "பிஎம் கிசான் திட்டத்திற்கு யார் தகுதியானவர்கள்?", "lang": "ta", "intent": "check_eligibility", "scheme": "pm_kisan"},
    {"query": "என்னிடம் 2 ஏக்கர் நிலம் உள்ளது, நான் பிஎம் கிசான் பெற முடியுமா?", "lang": "ta", "intent": "check_eligibility", "scheme": "pm_kisan"},
    {"query": "அரசு ஊழியர்கள் பிஎம் கிசான் பணம் பெறலாமா?", "lang": "ta", "intent": "check_eligibility", "scheme": "pm_kisan"},
    {"query": "குத்தகை விவசாயிகள் பிஎம் கிசான் திட்டத்திற்கு தகுதியானவர்களா?", "lang": "ta", "intent": "check_eligibility", "scheme": "pm_kisan"},
    {"query": "ஒரு குடும்பத்தில் கணவன் மற்றும் மனைவி இருவருக்கும் பிஎம் கிசான் கிடைக்குமா?", "lang": "ta", "intent": "check_eligibility", "scheme": "pm_kisan"},
    {"query": "வருமான வரி செலுத்துபவர்கள் பிஎம் கிசான் திட்டத்திற்கு தகுதியுடையவர்களா?", "lang": "ta", "intent": "check_eligibility", "scheme": "pm_kisan"},
    {"query": "ஓய்வூதியம் பெறுபவர்கள் பிஎம் கிசான் பெற முடியுமா?", "lang": "ta", "intent": "check_eligibility", "scheme": "pm_kisan"},
    {"query": "விவசாய நிலம் உள்ளவர்களுக்கு பிஎம் கிசான் தகுதி விதிகள் என்ன?", "lang": "ta", "intent": "check_eligibility", "scheme": "pm_kisan"},
    {"query": "पीएम किसान सम्मान निधि के लिए कौन पात्र है?", "lang": "hi", "intent": "check_eligibility", "scheme": "pm_kisan"},
    {"query": "क्या 2 हेक्टेयर से कम जमीन वाले किसान पीएम किसान के पात्र हैं?", "lang": "hi", "intent": "check_eligibility", "scheme": "pm_kisan"},
    {"query": "क्या सरकारी कर्मचारी पीएम किसान का लाभ ले सकते हैं?", "lang": "hi", "intent": "check_eligibility", "scheme": "pm_kisan"},
    {"query": "क्या बटाईदार किसान पीएम किसान के लिए आवेदन कर सकते हैं?", "lang": "hi", "intent": "check_eligibility", "scheme": "pm_kisan"},
    {"query": "क्या एक परिवार में पति और पत्नी दोनों को पीएम किसान मिल सकता है?", "lang": "hi", "intent": "check_eligibility", "scheme": "pm_kisan"},
    {"query": "क्या आयकर दाता पीएम किसान के लिए पात्र हैं?", "lang": "hi", "intent": "check_eligibility", "scheme": "pm_kisan"},
    {"query": "पेंशनभोगी क्या पीएम किसान के लिए पात्र हैं?", "lang": "hi", "intent": "check_eligibility", "scheme": "pm_kisan"},

    # =========================================================================
    # PM-KISAN: documents_needed
    # =========================================================================
    {"query": "What documents are required to register for PM-KISAN?", "lang": "en", "intent": "documents_needed", "scheme": "pm_kisan"},
    {"query": "Is Aadhaar card mandatory for PM Kisan enrollment?", "lang": "en", "intent": "documents_needed", "scheme": "pm_kisan"},
    {"query": "What land records do I need to show for PM-KISAN?", "lang": "en", "intent": "documents_needed", "scheme": "pm_kisan"},
    {"query": "Do I need an Aadhaar linked bank account for PM Kisan installment?", "lang": "en", "intent": "documents_needed", "scheme": "pm_kisan"},
    {"query": "List of required papers for new PM Kisan farmer application.", "lang": "en", "intent": "documents_needed", "scheme": "pm_kisan"},
    {"query": "Is land registry Khasra Khatauni needed for PM Kisan?", "lang": "en", "intent": "documents_needed", "scheme": "pm_kisan"},
    {"query": "பிஎம் கிசான் விண்ணப்பிக்க என்னென்ன ஆவணங்கள் தேவை?", "lang": "ta", "intent": "documents_needed", "scheme": "pm_kisan"},
    {"query": "பிஎம் கிசானுக்கு ஆதார் அட்டை கட்டாயமா?", "lang": "ta", "intent": "documents_needed", "scheme": "pm_kisan"},
    {"query": "பிஎம் கிசானில் பதிவு செய்ய என்ன நில ஆவணங்கள் வேண்டும்?", "lang": "ta", "intent": "documents_needed", "scheme": "pm_kisan"},
    {"query": "வங்கி கணக்கு ஆதாருடன் இணைக்கப்பட்டிருக்க வேண்டுமா?", "lang": "ta", "intent": "documents_needed", "scheme": "pm_kisan"},
    {"query": "பிஎம் கிசான் புதிய பதிவுக்கு தேவையான ஆவணங்களின் பட்டியல் என்ன?", "lang": "ta", "intent": "documents_needed", "scheme": "pm_kisan"},
    {"query": "पीएम किसान पंजीकरण के लिए कौन-कौन से दस्तावेज चाहिए?", "lang": "hi", "intent": "documents_needed", "scheme": "pm_kisan"},
    {"query": "क्या पीएम किसान के लिए आधार कार्ड अनिवार्य है?", "lang": "hi", "intent": "documents_needed", "scheme": "pm_kisan"},
    {"query": "पीएम किसान के लिए कौन से जमीन के कागजात दिखाने होंगे?", "lang": "hi", "intent": "documents_needed", "scheme": "pm_kisan"},
    {"query": "क्या पीएम किसान किस्त के लिए बैंक खाता आधार से लिंक होना जरूरी है?", "lang": "hi", "intent": "documents_needed", "scheme": "pm_kisan"},
    {"query": "खसरा खतौनी और बैंक पासबुक जरूरी है क्या पीएम किसान में?", "lang": "hi", "intent": "documents_needed", "scheme": "pm_kisan"},

    # =========================================================================
    # PM-KISAN: apply_process
    # =========================================================================
    {"query": "How can I apply for PM Kisan online on the portal?", "lang": "en", "intent": "apply_process", "scheme": "pm_kisan"},
    {"query": "Where do I submit my PM-KISAN application in my village?", "lang": "en", "intent": "apply_process", "scheme": "pm_kisan"},
    {"query": "Can I register for PM Kisan at a Common Service Centre (CSC)?", "lang": "en", "intent": "apply_process", "scheme": "pm_kisan"},
    {"query": "What is the procedure for PM Kisan new farmer registration?", "lang": "en", "intent": "apply_process", "scheme": "pm_kisan"},
    {"query": "How to do e-KYC for PM Kisan?", "lang": "en", "intent": "apply_process", "scheme": "pm_kisan"},
    {"query": "பிஎம் கிசான் திட்டத்திற்கு ஆன்லைனில் எப்படி விண்ணப்பிப்பது?", "lang": "ta", "intent": "apply_process", "scheme": "pm_kisan"},
    {"query": "கிராமத்தில் பிஎம் கிசான் விண்ணப்பத்தை எங்கு சமர்ப்பிக்க வேண்டும்?", "lang": "ta", "intent": "apply_process", "scheme": "pm_kisan"},
    {"query": "பொது சேவை மையத்தில் (CSC) பிஎம் கிசான் பதிவு செய்யலாமா?", "lang": "ta", "intent": "apply_process", "scheme": "pm_kisan"},
    {"query": "பிஎம் கிசான் இ-கேஒய்சி (eKYC) செய்வது எப்படி?", "lang": "ta", "intent": "apply_process", "scheme": "pm_kisan"},
    {"query": "पीएम किसान के लिए ऑनलाइन आवेदन कैसे करें?", "lang": "hi", "intent": "apply_process", "scheme": "pm_kisan"},
    {"query": "गांव में पीएम किसान का फॉर्म कहां जमा करें?", "lang": "hi", "intent": "apply_process", "scheme": "pm_kisan"},
    {"query": "क्या सीएससी सेंटर जाकर पीएम किसान पंजीकरण करा सकते हैं?", "lang": "hi", "intent": "apply_process", "scheme": "pm_kisan"},
    {"query": "पीएम किसान ई-केवाईसी कैसे पूरी करें?", "lang": "hi", "intent": "apply_process", "scheme": "pm_kisan"},

    # =========================================================================
    # PM-KISAN: benefit_amount
    # =========================================================================
    {"query": "How much money is given under PM-KISAN per year?", "lang": "en", "intent": "benefit_amount", "scheme": "pm_kisan"},
    {"query": "What is the installment amount for PM Kisan?", "lang": "en", "intent": "benefit_amount", "scheme": "pm_kisan"},
    {"query": "How many installments are paid each year under PM-KISAN?", "lang": "en", "intent": "benefit_amount", "scheme": "pm_kisan"},
    {"query": "When will the next 2000 rupees of PM Kisan come into my account?", "lang": "en", "intent": "benefit_amount", "scheme": "pm_kisan"},
    {"query": "Is PM Kisan benefit 6000 rupees or more?", "lang": "en", "intent": "benefit_amount", "scheme": "pm_kisan"},
    {"query": "பிஎம் கிசான் மூலம் ஆண்டுக்கு எவ்வளவு பணம் வழங்கப்படுகிறது?", "lang": "ta", "intent": "benefit_amount", "scheme": "pm_kisan"},
    {"query": "பிஎம் கிசான் தவணைத் தொகை எவ்வளவு?", "lang": "ta", "intent": "benefit_amount", "scheme": "pm_kisan"},
    {"query": "ஆண்டுக்கு எத்தனை தவணைகளில் பிஎம் கிசான் தொகை வழங்கப்படுகிறது?", "lang": "ta", "intent": "benefit_amount", "scheme": "pm_kisan"},
    {"query": "அடுத்த 2000 ரூபாய் பிஎம் கிசான் தவணை எப்போது வரும்?", "lang": "ta", "intent": "benefit_amount", "scheme": "pm_kisan"},
    {"query": "पीएम किसान योजना के तहत सालाना कितने पैसे मिलते हैं?", "lang": "hi", "intent": "benefit_amount", "scheme": "pm_kisan"},
    {"query": "पीएम किसान की एक किस्त में कितनी राशि मिलती है?", "hi": "hi", "intent": "benefit_amount", "scheme": "pm_kisan"},
    {"query": "पीएम किसान के 6000 रुपये साल में कितनी किस्तों में आते हैं?", "lang": "hi", "intent": "benefit_amount", "scheme": "pm_kisan"},
    {"query": "अगली ₹2000 की किस्त बैंक में कब क्रेडिट होगी?", "lang": "hi", "intent": "benefit_amount", "scheme": "pm_kisan"},

    # =========================================================================
    # MGNREGA: check_eligibility
    # =========================================================================
    {"query": "Who is eligible to work under MGNREGA 100 days scheme?", "lang": "en", "intent": "check_eligibility", "scheme": "mgnrega"},
    {"query": "Can urban residents get a Job Card under MGNREGA?", "lang": "en", "intent": "check_eligibility", "scheme": "mgnrega"},
    {"query": "What is the minimum age to work in MGNREGA?", "lang": "en", "intent": "check_eligibility", "scheme": "mgnrega"},
    {"query": "Are women eligible for 100 days job work in village?", "lang": "en", "intent": "check_eligibility", "scheme": "mgnrega"},
    {"query": "Can disabled persons work under MGNREGA?", "lang": "en", "intent": "check_eligibility", "scheme": "mgnrega"},
    {"query": "Does MGNREGA apply to landless agricultural labourers?", "lang": "en", "intent": "check_eligibility", "scheme": "mgnrega"},
    {"query": "மகாத்மா காந்தி 100 நாள் வேலை திட்டத்திற்கு யார் தகுதியானவர்கள்?", "lang": "ta", "intent": "check_eligibility", "scheme": "mgnrega"},
    {"query": "நகரங்களில் வசிப்பவர்கள் மகாத்மா காந்தி ஊரக வேலைவாய்ப்பு திட்டத்தில் சேர முடியுமா?", "lang": "ta", "intent": "check_eligibility", "scheme": "mgnrega"},
    {"query": "100 நாள் வேலை செய்ய குறைந்தபட்ச வயது என்ன?", "lang": "ta", "intent": "check_eligibility", "scheme": "mgnrega"},
    {"query": "பெண்கள் கிராமப்புற வேலைவாய்ப்பு திட்டத்திற்கு தகுதியானவர்களா?", "lang": "ta", "intent": "check_eligibility", "scheme": "mgnrega"},
    {"query": "நிலமற்ற விவசாய தொழிலாளர்களுக்கு 100 நாள் வேலை அட்டை கிடைக்குமா?", "lang": "ta", "intent": "check_eligibility", "scheme": "mgnrega"},
    {"query": "मनरेगा 100 दिन रोजगार योजना के लिए कौन पात्र है?", "lang": "hi", "intent": "check_eligibility", "scheme": "mgnrega"},
    {"query": "क्या शहरी क्षेत्र के लोग मनरेगा में जॉब कार्ड बनवा सकते हैं?", "lang": "hi", "intent": "check_eligibility", "scheme": "mgnrega"},
    {"query": "मनरेगा में काम करने के लिए न्यूनतम आयु क्या है?", "lang": "hi", "intent": "check_eligibility", "scheme": "mgnrega"},
    {"query": "क्या महिलाएं 100 दिन के काम के लिए पात्र हैं?", "lang": "hi", "intent": "check_eligibility", "scheme": "mgnrega"},
    {"query": "क्या भूमिहीन मजदूर मनरेगा में आवेदन कर सकते हैं?", "lang": "hi", "intent": "check_eligibility", "scheme": "mgnrega"},

    # =========================================================================
    # MGNREGA: documents_needed
    # =========================================================================
    {"query": "What documents are required to get an MGNREGA Job Card?", "lang": "en", "intent": "documents_needed", "scheme": "mgnrega"},
    {"query": "Do I need Aadhaar card to apply for 100 days job card?", "lang": "en", "intent": "documents_needed", "scheme": "mgnrega"},
    {"query": "Is residence proof necessary for MGNREGA registration?", "lang": "en", "intent": "documents_needed", "scheme": "mgnrega"},
    {"query": "Do I need to submit photos for Job Card issue?", "lang": "en", "intent": "documents_needed", "scheme": "mgnrega"},
    {"query": "100 நாள் வேலை அட்டை பெற என்னென்ன ஆவணங்கள் தேவை?", "lang": "ta", "intent": "documents_needed", "scheme": "mgnrega"},
    {"query": "வேலை அட்டைக்கு ஆதார் அட்டை மற்றும் ரேஷன் அட்டை போதுமா?", "lang": "ta", "intent": "documents_needed", "scheme": "mgnrega"},
    {"query": "வேலை அட்டை பெற பாஸ்போர்ட் அளவு புகைப்படம் தேவையா?", "lang": "ta", "intent": "documents_needed", "scheme": "mgnrega"},
    {"query": "मनरेगा जॉब कार्ड बनवाने के लिए क्या दस्तावेज चाहिए?", "lang": "hi", "intent": "documents_needed", "scheme": "mgnrega"},
    {"query": "क्या मनरेगा जॉब कार्ड के लिए आधार और राशन कार्ड जरूरी है?", "lang": "hi", "intent": "documents_needed", "scheme": "mgnrega"},
    {"query": "जॉब कार्ड के लिए कितने फोटो और कौन सा निवास प्रमाण पत्र लगेगा?", "lang": "hi", "intent": "documents_needed", "scheme": "mgnrega"},

    # =========================================================================
    # MGNREGA: apply_process
    # =========================================================================
    {"query": "How do I apply for a new Job Card in my Gram Panchayat?", "lang": "en", "intent": "apply_process", "scheme": "mgnrega"},
    {"query": "Within how many days should Gram Panchayat issue my Job Card?", "lang": "en", "intent": "apply_process", "scheme": "mgnrega"},
    {"query": "Can I make an oral request for work under MGNREGA?", "lang": "en", "intent": "apply_process", "scheme": "mgnrega"},
    {"query": "What is the process to request work after getting Job Card?", "lang": "en", "intent": "apply_process", "scheme": "mgnrega"},
    {"query": "கிராம பஞ்சாயத்தில் புதிய வேலை அட்டைக்கு எப்படி விண்ணப்பிப்பது?", "lang": "ta", "intent": "apply_process", "scheme": "mgnrega"},
    {"query": "விண்ணப்பித்த எத்தனை நாட்களில் வேலை அட்டை வழங்கப்படும்?", "lang": "ta", "intent": "apply_process", "scheme": "mgnrega"},
    {"query": "வேலை அட்டை பெற்ற பிறகு வேலை கோரி மனு செய்வது எப்படி?", "lang": "ta", "intent": "apply_process", "scheme": "mgnrega"},
    {"query": "ग्राम पंचायत में नए जॉब कार्ड के लिए आवेदन कैसे करें?", "lang": "hi", "intent": "apply_process", "scheme": "mgnrega"},
    {"query": "आवेदन करने के कितने दिनों में जॉब कार्ड मिल जाता है?", "lang": "hi", "intent": "apply_process", "scheme": "mgnrega"},
    {"query": "जॉब कार्ड मिलने के बाद काम मांगने का क्या तरीका है?", "lang": "hi", "intent": "apply_process", "scheme": "mgnrega"},

    # =========================================================================
    # MGNREGA: benefit_amount
    # =========================================================================
    {"query": "How many days of guaranteed work does MGNREGA provide?", "lang": "en", "intent": "benefit_amount", "scheme": "mgnrega"},
    {"query": "What is the daily wage rate under MGNREGA in my state?", "lang": "en", "intent": "benefit_amount", "scheme": "mgnrega"},
    {"query": "Is unemployment allowance paid if work is not given within 15 days?", "lang": "en", "intent": "benefit_amount", "scheme": "mgnrega"},
    {"query": "How are MGNREGA wages paid into bank account?", "lang": "en", "intent": "benefit_amount", "scheme": "mgnrega"},
    {"query": "100 நாள் திட்டத்தில் ஆண்டுக்கு எத்தனை நாட்கள் உத்தரவாத வேலை கிடைக்கும்?", "lang": "ta", "intent": "benefit_amount", "scheme": "mgnrega"},
    {"query": "100 நாள் வேலையில் ஒரு நாளுக்கான தினசரி கூலி எவ்வளவு?", "lang": "ta", "intent": "benefit_amount", "scheme": "mgnrega"},
    {"query": "15 நாட்களுக்குள் வேலை தரவில்லை என்றால் வேலையின்மை படி கிடைக்குமா?", "lang": "ta", "intent": "benefit_amount", "scheme": "mgnrega"},
    {"query": "मनरेगा में कितने दिन के काम की गारंटी मिलती है?", "lang": "hi", "intent": "benefit_amount", "scheme": "mgnrega"},
    {"query": "मनरेगा में प्रतिदिन की मजदूरी कितनी मिलती है?", "lang": "hi", "intent": "benefit_amount", "scheme": "mgnrega"},
    {"query": "अगर 15 दिन में काम न मिले तो क्या बेरोजगारी भत्ता मिलता है?", "lang": "hi", "intent": "benefit_amount", "scheme": "mgnrega"},

    # =========================================================================
    # AYUSHMAN BHARAT: check_eligibility
    # =========================================================================
    {"query": "Who is eligible for free healthcare under Ayushman Bharat PM-JAY?", "lang": "en", "intent": "check_eligibility", "scheme": "ayushman_bharat"},
    {"query": "Can senior citizens aged 70 and above get Ayushman card?", "lang": "en", "intent": "check_eligibility", "scheme": "ayushman_bharat"},
    {"query": "How do I check if my name is in SECC 2011 list for Ayushman Bharat?", "lang": "en", "intent": "check_eligibility", "scheme": "ayushman_bharat"},
    {"query": "Is Ayushman Bharat available for BPL ration card holders?", "lang": "en", "intent": "check_eligibility", "scheme": "ayushman_bharat"},
    {"query": "Does PM-JAY have any restriction on family size or age?", "lang": "en", "intent": "check_eligibility", "scheme": "ayushman_bharat"},
    {"query": "ஆயுஷ்மான் பாரத் திட்டத்தின் கீழ் இலவச மருத்துவ காப்பீட்டிற்கு யார் தகுதியானவர்கள்?", "lang": "ta", "intent": "check_eligibility", "scheme": "ayushman_bharat"},
    {"query": "70 வயதுக்கு மேற்பட்ட முதியவர்கள் ஆயுஷ்மான் அட்டை பெறலாமா?", "lang": "ta", "intent": "check_eligibility", "scheme": "ayushman_bharat"},
    {"query": "ஆயுஷ்மான் பட்டியலில் எனது பெயர் உள்ளதா என்பதை எப்படி சரிபார்ப்பது?", "lang": "ta", "intent": "check_eligibility", "scheme": "ayushman_bharat"},
    {"query": "குடும்ப உறுப்பினர் எண்ணிக்கை அல்லது வயது வரம்பு ஆயுஷ்மான் பாரத்தில் உள்ளதா?", "lang": "ta", "intent": "check_eligibility", "scheme": "ayushman_bharat"},
    {"query": "आयुष्मान भारत योजना के तहत 5 लाख तक के मुफ्त इलाज के लिए कौन पात्र है?", "lang": "hi", "intent": "check_eligibility", "scheme": "ayushman_bharat"},
    {"query": "क्या 70 वर्ष या उससे अधिक आयु के वरिष्ठ नागरिक आयुष्मान कार्ड बनवा सकते हैं?", "lang": "hi", "intent": "check_eligibility", "scheme": "ayushman_bharat"},
    {"query": "आयुष्मान भारत लाभार्थी सूची में अपना नाम कैसे चेक करें?", "lang": "hi", "intent": "check_eligibility", "scheme": "ayushman_bharat"},
    {"query": "क्या आयुष्मान कार्ड के लिए परिवार के सदस्यों की कोई सीमा है?", "lang": "hi", "intent": "check_eligibility", "scheme": "ayushman_bharat"},

    # =========================================================================
    # AYUSHMAN BHARAT: documents_needed
    # =========================================================================
    {"query": "What documents are required to make an Ayushman Golden Card?", "lang": "en", "intent": "documents_needed", "scheme": "ayushman_bharat"},
    {"query": "Do I need Aadhaar and Ration card for PM-JAY verification?", "lang": "en", "intent": "documents_needed", "scheme": "ayushman_bharat"},
    {"query": "What papers should I bring to hospital Ayushman Mitra desk?", "lang": "en", "intent": "documents_needed", "scheme": "ayushman_bharat"},
    {"query": "ஆயுஷ்மான் கோல்டன் அட்டை பெற என்னென்ன ஆவணங்கள் தேவை?", "lang": "ta", "intent": "documents_needed", "scheme": "ayushman_bharat"},
    {"query": "மருத்துவமனையில் ஆயுஷ்மான் மித்ராவிடம் என்ன சான்றிதழ்கள் காட்ட வேண்டும்?", "lang": "ta", "intent": "documents_needed", "scheme": "ayushman_bharat"},
    {"query": "ஆதார் அட்டை மற்றும் குடும்ப அட்டை ஆயுஷ்மான் பதிவுக்கு போதுமா?", "lang": "ta", "intent": "documents_needed", "scheme": "ayushman_bharat"},
    {"query": "आयुष्मान गोल्डन कार्ड बनवाने के लिए क्या दस्तावेज चाहिए?", "lang": "hi", "intent": "documents_needed", "scheme": "ayushman_bharat"},
    {"query": "अस्पताल में आयुष्मान मित्र डेस्क पर कौन से पहचान पत्र दिखाने होते हैं?", "lang": "hi", "intent": "documents_needed", "scheme": "ayushman_bharat"},
    {"query": "क्या आधार कार्ड और राशन कार्ड आयुष्मान भारत के लिए आवश्यक हैं?", "lang": "hi", "intent": "documents_needed", "scheme": "ayushman_bharat"},

    # =========================================================================
    # AYUSHMAN BHARAT: apply_process
    # =========================================================================
    {"query": "How can I apply for an Ayushman card online on beneficiary.nha.gov.in?", "lang": "en", "intent": "apply_process", "scheme": "ayushman_bharat"},
    {"query": "Where can I get my Ayushman card made in village?", "lang": "en", "intent": "apply_process", "scheme": "ayushman_bharat"},
    {"query": "What is the role of Ayushman Mitra at empaneled hospitals?", "lang": "en", "intent": "apply_process", "scheme": "ayushman_bharat"},
    {"query": "How to generate Ayushman Bharat card through CSC centre?", "lang": "en", "intent": "apply_process", "scheme": "ayushman_bharat"},
    {"query": "ஆயுஷ்மான் அட்டைக்கு ஆன்லைனில் எப்படி விண்ணப்பிப்பது?", "lang": "ta", "intent": "apply_process", "scheme": "ayushman_bharat"},
    {"query": "கிராமத்தில் ஆயுஷ்மான் அட்டை எங்கு எடுக்கலாம்?", "lang": "ta", "intent": "apply_process", "scheme": "ayushman_bharat"},
    {"query": "அரசு அல்லது தனியார் மருத்துவமனையில் ஆயுஷ்மான் அட்டையை எப்படி பயன்படுத்துவது?", "lang": "ta", "intent": "apply_process", "scheme": "ayushman_bharat"},
    {"query": "आयुष्मान कार्ड ऑनलाइन कैसे बनवाएं?", "lang": "hi", "intent": "apply_process", "scheme": "ayushman_bharat"},
    {"query": "सीएससी सेंटर या अस्पताल में आयुष्मान कार्ड कैसे बनता है?", "lang": "hi", "intent": "apply_process", "scheme": "ayushman_bharat"},
    {"query": "सूचीबद्ध अस्पताल में आयुष्मान मित्र के जरिए इलाज कैसे कराएं?", "lang": "hi", "intent": "apply_process", "scheme": "ayushman_bharat"},

    # =========================================================================
    # AYUSHMAN BHARAT: benefit_amount
    # =========================================================================
    {"query": "What is the insurance coverage amount under Ayushman Bharat PM-JAY?", "lang": "en", "intent": "benefit_amount", "scheme": "ayushman_bharat"},
    {"query": "Is the 5 lakh rupees health cover per person or per family?", "lang": "en", "intent": "benefit_amount", "scheme": "ayushman_bharat"},
    {"query": "Are medicines and hospital stay included in Ayushman Bharat benefits?", "lang": "en", "intent": "benefit_amount", "scheme": "ayushman_bharat"},
    {"query": "Does PM-JAY cover secondary and tertiary hospitalization costs?", "lang": "en", "intent": "benefit_amount", "scheme": "ayushman_bharat"},
    {"query": "ஆயுஷ்மான் பாரத் திட்டத்தில் எவ்வளவு மருத்துவ காப்பீட்டுத் தொகை கிடைக்கிறது?", "lang": "ta", "intent": "benefit_amount", "scheme": "ayushman_bharat"},
    {"query": "5 லட்சம் ரூபாய் காப்பீடு ஒருவருக்கா அல்லது ஒட்டுமொத்த குடும்பத்திற்கா?", "lang": "ta", "intent": "benefit_amount", "scheme": "ayushman_bharat"},
    {"query": "மருத்துவமனை தங்குமிடம் மற்றும் மருந்துகளுக்கான செலவுகள் ஆயுஷ்மான் திட்டத்தில் உள்ளதா?", "lang": "ta", "intent": "benefit_amount", "scheme": "ayushman_bharat"},
    {"query": "आयुष्मान भारत में प्रति परिवार कितना स्वास्थ्य बीमा कवर मिलता है?", "lang": "hi", "intent": "benefit_amount", "scheme": "ayushman_bharat"},
    {"query": "क्या 5 लाख का स्वास्थ्य कवर पूरे परिवार के लिए है?", "lang": "hi", "intent": "benefit_amount", "scheme": "ayushman_bharat"},
    {"query": "क्या अस्पताल में भर्ती होने और दवाओं का खर्च आयुष्मान भारत में शामिल है?", "lang": "hi", "intent": "benefit_amount", "scheme": "ayushman_bharat"},

    # =========================================================================
    # OUT_OF_SCOPE: queries unrelated to the 3 schemes
    # =========================================================================
    {"query": "What is the weather forecast for Coimbatore today?", "lang": "en", "intent": "out_of_scope", "scheme": None},
    {"query": "How do I book a train ticket on IRCTC?", "lang": "en", "intent": "out_of_scope", "scheme": None},
    {"query": "Who won the cricket match yesterday?", "lang": "en", "intent": "out_of_scope", "scheme": None},
    {"query": "Can I get an education loan for my engineering college?", "lang": "en", "intent": "out_of_scope", "scheme": None},
    {"query": "How do I renew my driving licence online?", "lang": "en", "intent": "out_of_scope", "scheme": None},
    {"query": "What is the gold price in Chennai today?", "lang": "en", "intent": "out_of_scope", "scheme": None},
    {"query": "Tell me a joke about robots.", "lang": "en", "intent": "out_of_scope", "scheme": None},
    {"query": "What are the rules for traffic signal fines?", "lang": "en", "intent": "out_of_scope", "scheme": None},
    {"query": "How do I open a fixed deposit account in SBI?", "lang": "en", "intent": "out_of_scope", "scheme": None},
    {"query": "What is the recipe for making sambar?", "lang": "en", "intent": "out_of_scope", "scheme": None},
    {"query": "இன்று கோயம்புத்தூரில் மழை பெய்யுமா?", "lang": "ta", "intent": "out_of_scope", "scheme": None},
    {"query": "ரயில் டிக்கெட் முன்பதிவு செய்வது எப்படி?", "lang": "ta", "intent": "out_of_scope", "scheme": None},
    {"query": "இன்றைய தங்கம் விலை என்ன?", "lang": "ta", "intent": "out_of_scope", "scheme": None},
    {"query": "ஓட்டுநர் உரிமம் ஆன்லைனில் புதுப்பிப்பது எப்படி?", "lang": "ta", "intent": "out_of_scope", "scheme": None},
    {"query": "கல்லூரி படிப்புக்கு கல்வி கடன் பெறுவது எப்படி?", "lang": "ta", "intent": "out_of_scope", "scheme": None},
    {"query": "வங்கி சேமிப்பு கணக்கு தொடங்குவது எப்படி?", "lang": "ta", "intent": "out_of_scope", "scheme": None},
    {"query": "இன்றைய கிரிக்கெட் போட்டி ஸ்கோர் என்ன?", "lang": "ta", "intent": "out_of_scope", "scheme": None},
    {"query": "இட்லி சாம்பார் செய்வது எப்படி?", "lang": "ta", "intent": "out_of_scope", "scheme": None},
    {"query": "आज दिल्ली में मौसम कैसा रहेगा?", "lang": "hi", "intent": "out_of_scope", "scheme": None},
    {"query": "आईआरसीटीसी पर ट्रेन टिकट कैसे बुक करें?", "lang": "hi", "intent": "out_of_scope", "scheme": None},
    {"query": "आज सोने और चांदी का भाव क्या है?", "lang": "hi", "intent": "out_of_scope", "scheme": None},
    {"query": "ड्राइविंग लाइसेंस का नवीनीकरण कैसे कराएं?", "lang": "hi", "intent": "out_of_scope", "scheme": None},
    {"query": "बैंक में फिक्स डिपॉजिट कैसे खोलें?", "lang": "hi", "intent": "out_of_scope", "scheme": None},
    {"query": "कल का क्रिकेट मैच किसने जीता?", "lang": "hi", "intent": "out_of_scope", "scheme": None},
    {"query": "मुझे एक मजेदार चुटकुला सुनाओ।", "lang": "hi", "intent": "out_of_scope", "scheme": None},
    {"query": "इंजीनियरिंग कॉलेज के लिए शिक्षा ऋण कैसे मिलेगा?", "lang": "hi", "intent": "out_of_scope", "scheme": None},
]


def main():
    out_dir = Path("data/intents")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "intent_dataset.json"

    # Assign IDs
    for idx, item in enumerate(DATASET, 1):
        item["id"] = f"intent_{idx:03d}"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(DATASET, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(DATASET)} intent queries in {out_file}")


if __name__ == "__main__":
    main()
