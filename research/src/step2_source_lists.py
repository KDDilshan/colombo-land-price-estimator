"""Step 2 - Source lists for the first four amenity categories.

Each list is a verbatim transcription of a named source register.  Nothing is
added, nothing is trimmed to match the original study's counts, and no entry
is derived from OpenStreetMap tags.  Membership of the source list is the only
rule; geographic membership is decided later by point-in-polygon.

Entries are (name, locality) - locality is used only to disambiguate the
geocoder query, never to decide district membership.
"""

# ---------------------------------------------------------------------------
# 1. EXPRESSWAY ENTRANCES
# Source: RDA Expressway Operation Maintenance & Management Division, toll
# tariff schedule image (Gazette Extraordinary 2467/49, 2025-12-17) plus the
# EOM&M homepage notice of 2025-12-22 for Seeduwa.
# Final set fixed by the user: Colombo district plus 10 km buffer candidates.
# ---------------------------------------------------------------------------
EXPRESSWAY_ENTRANCES = [
    ("Kottawa Interchange", "Kottawa"),
    ("Kahathuduwa Interchange", "Kahathuduwa"),
    ("Gelanigama Interchange", "Gelanigama, Bandaragama"),
    ("Athurugiriya Interchange", "Athurugiriya"),
    ("Kotalawala Interchange", "Kotalawala, Kaduwela"),
    ("Kaduwela Interchange", "Kaduwela"),
    ("Kadawatha Interchange", "Kadawatha"),
    ("Peliyagoda Interchange", "Peliyagoda"),
    ("Kerawalapitiya Interchange KIC2", "Kerawalapitiya, Wattala"),
    ("Seeduwa Interchange", "Seeduwa"),
    ("Ja-Ela Interchange", "Ja-Ela"),
]

# ---------------------------------------------------------------------------
# 2. GOVERNMENT HOSPITALS
# Source: Ministry of Health institution directory,
# health.gov.lk/health-institutions-in-sri-lanka/ retrieved 2026-08-02.
# Enumerated by institution TYPE with NO district filter, because the
# directory's district field is a health region and is blank entirely for the
# specialised national institutes.  Divisional hospitals were verified to have
# a fully populated district field (per-district counts sum to 467 = the
# national total), so they are taken for Colombo and its four adjacent
# districts only - a hospital in a non-adjacent district cannot lie within
# 10 km of the Colombo boundary.
# ---------------------------------------------------------------------------
HOSP_TEACHING = [
    ("Teaching Hospital Anuradhapura", "Anuradhapura"),
    ("Teaching Hospital Batticaloa", "Batticaloa"),
    ("Castle Street Hospital for Women", "Colombo 08"),
    ("Colombo South Teaching Hospital", "Kalubowila"),
    ("Colombo North Teaching Hospital", "Ragama"),
    ("De Soysa Maternity Hospital for Women", "Colombo 08"),
    ("Dental Hospital Peradeniya", "Peradeniya"),
    ("Dental Institute Colombo", "Colombo 07"),
    ("Teaching Hospital Jaffna", "Jaffna"),
    ("Teaching Hospital Karapitiya", "Karapitiya, Galle"),
    ("Lady Ridgeway Hospital for Children", "Colombo 08"),
    ("Teaching Hospital Mahamodara", "Mahamodara, Galle"),
    ("National Cancer Institute Apeksha Hospital", "Maharagama"),
    ("Teaching Hospital Kuliyapitiya", "Kuliyapitiya"),
    ("Teaching Hospital Kurunegala", "Kurunegala"),
    ("Teaching Hospital Ratnapura", "Ratnapura"),
    ("National Institute of Mental Health", "Mulleriyawa New Town, Angoda"),
    ("National Eye Hospital", "Deans Road, Colombo 10"),
    ("Teaching Hospital Peradeniya", "Peradeniya"),
    ("Sirimavo Bandaranayake Specialized Childrens Hospital", "Peradeniya"),
]

HOSP_NATIONAL = [
    ("National Hospital of Sri Lanka", "Colombo"),
    ("National Hospital Kandy", "Kandy"),
]

HOSP_SPECIALIZED = [
    ("National Hospital for Respiratory Diseases", "Welisara"),
    ("Leprosy Hospital Hendala", "Hendala"),
    ("Leprosy Hospital Manthivu", "Batticaloa"),
    ("Rheumatology and Rehabilitation Hospital Ragama", "Ragama"),
]

HOSP_GENERAL = [
    ("District General Hospital Embilipitiya", "Embilipitiya"),
    ("District General Hospital Gampaha", "Gampaha"),
    ("District General Hospital Negombo", "Negombo"),
]

HOSP_PROVINCIAL_GENERAL = [
    ("Provincial General Hospital Badulla", "Badulla"),
]

HOSP_DISTRICT_GENERAL = [
    ("District General Hospital Ampara", "Ampara"),
    ("District General Hospital Chilaw", "Chilaw"),
    ("District General Hospital Hambantota", "Hambantota"),
    ("District General Hospital Kalutara", "Kalutara"),
    ("District General Hospital Kegalle", "Kegalle"),
    ("District General Hospital Matara", "Matara"),
    ("District General Hospital Monaragala", "Monaragala"),
    ("District General Hospital Nuwara Eliya", "Nuwara Eliya"),
    ("District General Hospital Polonnaruwa", "Polonnaruwa"),
    ("District General Hospital Trincomalee", "Trincomalee"),
    ("District General Hospital Matale", "Matale"),
    ("District General Hospital Kilinochchi", "Kilinochchi"),
    ("District General Hospital Mannar", "Mannar"),
    ("District General Hospital Mullaitivu", "Mullaitivu"),
    ("District General Hospital Vavuniya", "Vavuniya"),
]

HOSP_OTHER = [
    ("Divisional Hospital Aluthgama", "Aluthgama"),
    ("Divisional Hospital Darga Town", "Darga Town"),
    ("Divisional Hospital Kandana", "Kandana"),
    ("Prison Hospital Mahara", "Mahara"),
    ("Prison Hospital Welikade", "Welikada, Colombo"),
    ("Police Hospital Narahenpita", "Narahenpita, Colombo"),
]

HOSP_BOARD_MANAGED = [
    ("Wijaya Kumaranatunga Memorial Hospital", "Colombo"),
    ("Sri Jayawardenapura General Hospital", "Sri Jayawardenapura, Nugegoda"),
    ("Neville Fernando Hospital Malabe", "Malabe"),
]

HOSP_BASE = [
    ("Base Hospital Kebithigollewa", "Kebithigollewa"),
    ("Base Hospital Padaviya", "Padaviya"),
    ("Base Hospital Thambuththegama", "Thambuththegama"),
    ("Base Hospital Medirigiriya", "Medirigiriya"),
    ("Base Hospital Walikanda", "Walikanda"),
    ("Base Hospital Hingurakgoda", "Hingurakgoda"),
    ("Base Hospital Dambadeniya", "Dambadeniya"),
    ("Base Hospital Galgamuwa", "Galgamuwa"),
    ("Base Hospital Nikaweratiya", "Nikaweratiya"),
    ("Base Hospital Polpithigama", "Polpithigama"),
    ("Base Hospital Marawila", "Marawila"),
    ("Base Hospital Puttalam", "Puttalam"),
    ("Base Hospital Anamaduwa", "Anamaduwa"),
    ("Base Hospital Kalpitiya", "Kalpitiya"),
    ("Base Hospital Akkaraipattu", "Akkaraipattu"),
    ("Base Hospital Angoda National Institute of Infectious Diseases", "Angoda"),
    ("Base Hospital Kalmunai South Ashroff Memorial Hospital", "Kalmunai South"),
    ("Base Hospital Gampola", "Gampola"),
    ("Base Hospital Kalmunai North", "Kalmunai North"),
    ("Base Hospital Kanthale", "Kanthalai"),
    ("Colombo East Base Hospital Mulleriyawa", "Mulleriyawa"),
    ("Base Hospital Beruwala", "Beruwala"),
    ("Base Hospital Karawanella", "Karawanella"),
    ("Base Hospital Mawanella", "Mawanella"),
    ("Base Hospital Warakapola", "Warakapola"),
    ("Base Hospital Balangoda", "Balangoda"),
    ("Base Hospital Kahawatta", "Kahawatta"),
    ("Base Hospital Kalawana", "Kalawana"),
    ("Base Hospital Ehaliyagoda", "Eheliyagoda"),
    ("Base Hospital Kolonna", "Kolonna"),
    ("Base Hospital Balapitiya", "Balapitiya"),
    ("Base Hospital Elpitiya", "Elpitiya"),
    ("Base Hospital Udugama", "Udugama"),
    ("Divisional Hospital Induruwa", "Induruwa"),
    ("Base Hospital Bibile", "Bibile"),
    ("Base Hospital Siyambalanduwa", "Siyambalanduwa"),
    ("Base Hospital Wellawaya", "Wellawaya"),
    ("Base Hospital Tangalle", "Tangalle"),
    ("Base Hospital Tissamaharama", "Tissamaharama"),
    ("Base Hospital Walasmulla", "Walasmulla"),
    ("Base Hospital Deniyaya", "Deniyaya"),
    ("Base Hospital Kamburupitiya", "Kamburupitiya"),
    ("Base Hospital Diyatalawa", "Diyatalawa"),
    ("Base Hospital Mahiyanganaya", "Mahiyanganaya"),
    ("Base Hospital Welimada", "Welimada"),
    ("Base Hospital Minuwangoda", "Minuwangoda"),
    ("Base Hospital Avissawella", "Avissawella"),
    ("Base Hospital Homagama", "Homagama"),
    ("Base Hospital Horana", "Horana"),
    ("Base Hospital Panadura", "Panadura"),
    ("Kethumathi Maternity Hospital", "Panadura"),
    ("Base Hospital Pimbura", "Pimbura, Agalawatta"),
    ("Base Hospital Kiribathgoda", "Kiribathgoda"),
    ("Base Hospital Mirigama", "Mirigama"),
    ("Base Hospital Wathupitiwala", "Wathupitiwala"),
    ("Base Hospital Teldeniya", "Teldeniya"),
    ("Base Hospital Dambulla", "Dambulla"),
    ("Base Hospital Dickoya", "Dickoya"),
    ("Base Hospital Rikillagaskada", "Rikillagaskada"),
    ("Base Hospital Kinniya", "Kinniya"),
    ("Base Hospital Muttur", "Muttur"),
    ("Base Hospital Pullmodai", "Pulmoddai"),
    ("Base Hospital Dehiatthakandiya", "Dehiattakandiya"),
    ("Base Hospital Mahaoya", "Mahaoya"),
    ("Base Hospital Eravur", "Eravur"),
    ("Base Hospital Kaluwanchikudy", "Kaluwanchikudy"),
    ("Base Hospital Kattankudy", "Kattankudy"),
    ("Base Hospital Valaichchenai", "Valaichchenai"),
    ("Base Hospital Pottuvil", "Pottuvil"),
    ("Base Hospital Sammanthurai", "Sammanthurai"),
    ("Base Hospital Ninthavur", "Ninthavur"),
    ("Base Hospital Thirukkovil", "Thirukkovil"),
    ("Base Hospital Point Pedro", "Point Pedro"),
    ("Base Hospital Tellippalai", "Tellippalai"),
    ("Base Hospital Chavakachcheri", "Chavakachcheri"),
    ("Base Hospital Kayts", "Kayts"),
    ("Base Hospital Mulankavil", "Mulankavil"),
    ("Base Hospital Murungan", "Murunkan"),
    ("Base Hospital Mankulam", "Mankulam"),
    ("Base Hospital Mallavi", "Mallavi"),
    ("Base Hospital Puthukkuduiyiruppu", "Puthukkudiyiruppu"),
    ("Base Hospital Cheddikulam", "Cheddikulam"),
    ("Base Hospital Medawachchiya", "Medawachchiya"),
    ("Base Hospital Kekirawa", "Kekirawa"),
    ("Base Hospital Kahatagasdigiliya", "Kahatagasdigiliya"),
]

# Divisional hospitals - Colombo and its four adjacent districts only.
HOSP_DIVISIONAL = [
    # Colombo
    ("Divisional Hospital Athurugiriya", "Athurugiriya"),
    ("Divisional Hospital Piliyandala", "Piliyandala"),
    ("Divisional Hospital Colombo Central", "Maligawatte, Colombo"),
    ("Divisional Hospital Salawa", "Salawa, Kosgama"),
    ("Divisional Hospital Moratuwa", "Moratuwa"),
    ("Divisional Hospital Talangama", "Thalangama, Koswatta"),
    ("Divisional Hospital Nawagamuwa", "Nawagamuwa, Ranala"),
    ("Divisional Hospital Wethara", "Wethara, Polgasowita"),
    ("Divisional Hospital Padukka", "Padukka"),
    # Gampaha
    ("Divisional Hospital Akaragama", "Akaragama"),
    ("Divisional Hospital Malwathuhiripitiya", "Malwathuhiripitiya"),
    ("Divisional Hospital Biyagama", "Biyagama"),
    ("Divisional Hospital Bokalagama", "Bokalagama"),
    ("Divisional Hospital Pamunugama", "Pamunugama"),
    ("Divisional Hospital Divulapitiya", "Divulapitiya"),
    ("Divisional Hospital Radawana", "Radawana"),
    ("Divisional Hospital Dompe", "Dompe"),
    ("Divisional Hospital Udupila", "Udupila"),
    ("Divisional Hospital Ja-Ela", "Ja-Ela"),
    # Kalutara
    ("Divisional Hospital Dodangoda", "Dodangoda"),
    ("Divisional Hospital Haltota", "Haltota"),
    ("Divisional Hospital Baduraliya", "Baduraliya"),
    ("Divisional Hospital Ingiriya", "Ingiriya"),
    ("Divisional Hospital Bandaragama", "Bandaragama"),
    ("Divisional Hospital Ittepana", "Ittepana"),
    ("Divisional Hospital Neboda", "Neboda"),
    ("Divisional Hospital Katugahahena", "Katugahahena"),
    ("Divisional Hospital Bulathsinhala", "Bulathsinhala"),
    ("Divisional Hospital Mathugama", "Mathugama"),
    ("Divisional Hospital Galpatha", "Galpatha"),
    ("Divisional Hospital Meegahatenne", "Meegahatenna"),
    ("Divisional Hospital Morontuduwa", "Morontuduwa"),
    # Kegalle
    ("Divisional Hospital Amitirigala", "Amitirigala"),
    ("Divisional Hospital Halgolla", "Halgolla"),
    ("Divisional Hospital Aranayaka", "Aranayaka"),
    ("Divisional Hospital Hemmathagama", "Hemmathagama"),
    ("Divisional Hospital Beligala", "Beligala"),
    ("Divisional Hospital Hinguralakanda", "Hinguralakanda"),
    ("Divisional Hospital Dedugala", "Dedugala"),
    ("Divisional Hospital Kithulgala", "Kitulgala"),
    ("Divisional Hospital Dematanpitiya", "Dematanpitiya"),
    ("Divisional Hospital Mahapallegama", "Mahapallegama"),
    ("Divisional Hospital Deraniyagala", "Deraniyagala"),
    ("Divisional Hospital Pindeniya", "Pindeniya"),
    ("Divisional Hospital ERH Kiriporuwa", "Kiriporuwa"),
    ("Divisional Hospital Rambukkana", "Rambukkana"),
    ("Divisional Hospital Ganthuna", "Ganthuna"),
    ("Divisional Hospital Sapumalkanda", "Sapumalkanda"),
    ("Divisional Hospital Gonagaldeniya", "Gonagaldeniya"),
    ("Divisional Hospital Undugoda", "Undugoda"),
    # Ratnapura
    ("Divisional Hospital Downside", "Downside, Ratnapura"),
    ("Divisional Hospital Alupola", "Alupola"),
    ("Divisional Hospital Mahawalatenna", "Mahawalatenna"),
    ("Divisional Hospital Ayagama", "Ayagama"),
    ("Divisional Hospital Marathenna", "Marathenna"),
    ("Divisional Hospital Belihuloya", "Belihuloya"),
    ("Divisional Hospital Nivitigala", "Nivitigala"),
    ("Divisional Hospital Chandrikawewa", "Chandrikawewa"),
    ("Divisional Hospital Omalpe", "Omalpe"),
    ("Divisional Hospital Dumbara", "Dumbara, Ratnapura"),
    ("Divisional Hospital Pallebedda", "Pallebedda"),
    ("Divisional Hospital Endana", "Endana"),
    ("Divisional Hospital Pelmadulla", "Pelmadulla"),
    ("Divisional Hospital Erathna", "Erathna"),
    ("Divisional Hospital Pothupitiya", "Pothupitiya, Ratnapura"),
    ("Divisional Hospital Gallella", "Gallella"),
    ("Divisional Hospital Rakwana", "Rakwana"),
    ("Divisional Hospital Gilimale", "Gilimale"),
    ("Divisional Hospital Ranwala", "Ranwala"),
    ("Divisional Hospital Godakawela", "Godakawela"),
    ("Divisional Hospital Rassagala", "Rassagala"),
    ("Divisional Hospital Hunuwala", "Hunuwala"),
    ("Divisional Hospital Sooriyakanda", "Sooriyakanda"),
    ("Divisional Hospital Kaltota", "Kaltota"),
    ("Divisional Hospital Teppanawa", "Teppanawa"),
    ("Divisional Hospital Kiribathgala", "Kiribathgala"),
    ("Divisional Hospital Udawalawa", "Udawalawe"),
    ("Divisional Hospital Kiriella", "Kiriella"),
    ("Divisional Hospital Weligepola", "Weligepola"),
    ("Divisional Hospital Kiriporuwa", "Kiriporuwa, Ratnapura"),
    ("Divisional Hospital Madampe", "Madampe, Ratnapura"),
    ("Divisional Hospital Palam Kotte", "Palam Kotte"),
]

GOVT_HOSPITALS = {
    "Teaching Hospitals": HOSP_TEACHING,
    "National Hospitals": HOSP_NATIONAL,
    "Specialized Hospitals": HOSP_SPECIALIZED,
    "General Hospital": HOSP_GENERAL,
    "Provincial General Hospitals": HOSP_PROVINCIAL_GENERAL,
    "District General Hospitals": HOSP_DISTRICT_GENERAL,
    "Other Hospitals": HOSP_OTHER,
    "Board Managed Hospitals": HOSP_BOARD_MANAGED,
    "Base Hospitals": HOSP_BASE,
    "Divisional Hospitals": HOSP_DIVISIONAL,
}

# ---------------------------------------------------------------------------
# 3. UNIVERSITIES
# Source: University Grants Commission register, ugc.ac.lk, retrieved
# 2026-08-02 via the browser (the com_university component returns HTTP 500 to
# a plain fetch but renders).  Set fixed by the user as Universities (17) +
# Campuses (2) + Other Government Universities (6) = 25, the only combination
# reaching the original study's figure.  The 18 UGC Institutes are excluded.
# ---------------------------------------------------------------------------
UNI_UNIVERSITIES = [
    ("University of Colombo", "Colombo 03"),
    ("University of Peradeniya", "Peradeniya"),
    ("University of Sri Jayewardenepura", "Gangodawila, Nugegoda"),
    ("University of Kelaniya", "Dalugama, Kelaniya"),
    ("University of Moratuwa", "Katubedda, Moratuwa"),
    ("University of Jaffna", "Jaffna"),
    ("University of Ruhuna", "Matara"),
    ("The Open University of Sri Lanka", "Nawala, Nugegoda"),
    ("Eastern University Sri Lanka", "Vantharumoolai, Chenkalady"),
    ("South Eastern University of Sri Lanka", "Oluvil"),
    ("Rajarata University of Sri Lanka", "Mihintale"),
    ("Sabaragamuwa University of Sri Lanka", "Belihuloya"),
    ("Wayamba University of Sri Lanka", "Kuliyapitiya"),
    ("Uva Wellassa University of Sri Lanka", "Badulla"),
    ("University of the Visual and Performing Arts", "Colombo 07"),
    ("Gampaha Wickramarachchi University of Indigenous Medicine", "Yakkala"),
    ("University of Vavuniya", "Vavuniya"),
]
UNI_CAMPUSES = [
    ("Sri Palee Campus University of Colombo", "Horana"),
    ("Trincomalee Campus Eastern University", "Trincomalee"),
]
UNI_OTHER_GOVT = [
    ("General Sir John Kotelawala Defence University", "Ratmalana"),
    ("Buddhist and Pali University of Sri Lanka", "Pitipana, Homagama"),
    ("Bhiksu University of Sri Lanka", "Anuradhapura"),
    ("University of Vocational Technology", "Ratmalana"),
    ("Ocean University of Sri Lanka", "Mattakkuliya, Colombo 15"),
    ("National Institute of Education", "Maharagama"),
]
UNIVERSITIES = {
    "Universities": UNI_UNIVERSITIES,
    "Campuses": UNI_CAMPUSES,
    "Other Government Universities": UNI_OTHER_GOVT,
}

# ---------------------------------------------------------------------------
# 4. INTERNATIONAL SCHOOLS
# Source: TISSL member list.  The live site renders "No members found!", so the
# list is taken from the Wayback Machine snapshot 20220117102815 of
# tissl.lk/members.php - inside the original study period.  One row per
# CAMPUS, not per membership, because the variable measures distance to a
# physical school.
# ---------------------------------------------------------------------------
INTL_SCHOOLS = [
    ("Alethea International School", "13 Sri Mahabodhi Road, Dehiwela"),
    ("Asian International School", "4/97 Thalakotuwa Gardens, Colombo 5"),
    ("Belvoir College International", "20 Lester James Peiris Mawatha, Colombo 5"),
    ("Burhani Serendib School", "41 Glen Aber Place, Colombo 4"),
    ("Colombo International School Colombo", "28 Gregorys Road, Colombo 7"),
    ("Colombo International School Kandy", "175 Paranagantota Road, Mawilmada, Kandy"),
    ("Elizabeth Moir School Senior", "4/20 Thalakotuwa Gardens, Colombo 5"),
    ("Elizabeth Moir School Junior", "100 Park Road, Colombo 5"),
    ("Gateway College Colombo", "185 Koswatta Road, Rajagiriya"),
    ("Gateway College Kandy", "80 Wariyapola Sri Sumangala Mawatha, Asgiriya, Kandy"),
    ("Ilma International Girls School", "4/100 Thalakotuwa Road, Colombo 5"),
    ("Leeds International School Panadura", "105 Arthur V Dias Mawatha, Panadura"),
    ("Lyceum International School Nugegoda", "3/1 Raymond Road, Nugegoda"),
    ("Lyceum International School Wattala", "32 Royal Pearl Gardens, Hendala, Wattala"),
    ("Lyceum International School Panadura", "8/3A Arthur V Dias Mawatha, Walana, Panadura"),
    ("Lyceum International School Ratnapura", "115 Sripada Mawatha, Ratnapura"),
    ("OKI International School", "43 Old Negombo Road, Wattala"),
    ("Wycherley International School Gampaha", "5 Mudungoda, Miriswaththa, Gampaha"),
    ("Royal Institute", "189 Havelock Road, Colombo 05"),
    ("Stafford International School", "37 Guildford Crescent, Colombo 7"),
    ("The British School in Colombo", "63 Elvitigala Mawatha, Colombo 08"),
    ("The Overseas School of Colombo", "Pelawatte, Battaramulla"),
    ("Wycherley International School Senior", "5 Queens Road, Colombo 3"),
    ("Wycherley International School Junior", "232 Bauddhaloka Mawatha, Colombo 7"),
    ("Horizon College International", "482/B Millennium Drive, Malabe"),
]
