"""Default prompts for the document classifier."""

# --- Shared Rules (used by both OpenAI and Ollama) ---

RULES_TITLE = """TITEL-REGELN:
- Der Titel ist das WICHTIGSTE Feld -- er muss das Dokument eindeutig identifizierbar machen
- NICHT einfach Dokumenttyp + Korrespondent wiederholen! Diese stehen schon in eigenen Feldern
- Stattdessen: WAS ist der konkrete INHALT/GEGENSTAND des Dokuments?
- Maximal 8-10 Worte, keine ganzen Saetze

REFERENZNUMMERN -- NUR diese Typen in den Titel aufnehmen:
- Explizit als Rechnungs-Nr., Auftragsnr., Vertragsnr., Aktenzeichen bezeichnete Nummern
- NICHT: Personalnummern, Mitarbeiternummern, Kundennummern, eTIN, Steuernummern,
         Finanzamtsnummern, Formular-Nummern, Feld-Nummern, IBAN, Steuerklassen
- Wenn keine eindeutige Dokumentreferenz erkennbar ist: KEINE Nummer erfinden oder rate!

INHALT JE DOKUMENTTYP:
- Rechnungen/Lieferscheine: Wofuer? (Dienstleistung, Produkt, Zeitraum) + Rechnungsnr. wenn vorhanden
- Vertraege: Art des Vertrags (Mietvertrag, Kaufvertrag) + Gegenstand
- Versicherungen: Was ist versichert? (KFZ, Haftpflicht, Hausrat)
- Bescheide/Steuerdokumente: Dokumenttyp + Steuerjahr (z.B. "Lohnsteuerbescheinigung 2015", "Einkommensteuerbescheid 2023")
- Gehaltsabrechnungen: Monat + Jahr (z.B. "Gehaltsabrechnung Maerz 2015")
- Kontoauszuege: Monat/Zeitraum (z.B. "Kontoauszug Januar 2024")
- Angebote: Was wird angeboten? (Leistung, Produkt)

- KEIN "Dokument", "PDF", "Scan" im Titel
- Beispiele GUTER Titel (Muster, NICHT wortwortlich kopieren!):
  * "Rechnung RE-2024-815 Gasverbrauch Quartal 3"
  * "Lohnsteuerbescheinigung 2015" (kein Personalausweis-Nr.!)
  * "Mietvertrag Kastanienweg 21 Willich"
  * "Gehaltsabrechnung Oktober 2023"
- Beispiele SCHLECHTER Titel:
  * "Lohnsteuerbescheinigung 2024-03557" -- 03557 ist Personalnummer, NICHT Dokumentreferenz!
  * "[Dokumenttyp] [Firmenname]" -- WAS fehlt!
  * Zufaellige Zahlen aus dem Dokument als vermeintliche Referenz verwenden"""

RULES_CORRESPONDENT = """KORRESPONDENT-REGELN:
- Der Absender/Aussteller/die Firma die das Dokument erstellt hat
- NUR ein einziger Korrespondent
{% if TRIM_PROMPT %}
- KURZNAME: Verwende NUR den Kernmarkennamen ohne Rechtsform-Zusaetze!
  Beispiele: "Telekom" statt "Deutsche Telekom AG", "IKEA" statt "IKEA Deutschland GmbH & Co. KG",
             "Sparkasse" statt "Stadtsparkasse Muenchen", "AOK" statt "AOK Bayern GmbH",
             "Allianz" statt "Allianz Versicherungs-AG"
- KEINE Rechtsformen im Namen (GmbH, AG, KG, GmbH & Co. KG, UG, OHG, Ltd., Inc., SE, eV, ...)
- KEIN Laender-Praefix wenn der Kurzname allgemein bekannt ist (z.B. nicht "Deutsche X" sondern "X")
{% else %}
- Den Namen so uebernehmen wie er im lesbaren Text steht (Absenderzeile, Fusszeile, Unterschrift)
{% endif %}
- WICHTIG: Nur angeben wenn du dir SICHER bist! Das OCR eines Logos oder Briefkopfs kann unleserlich sein.
- Bei Unsicherheit: null zurueckgeben -- NIEMALS raten oder erfinden!
- Den Korrespondenten NUR aus dem tatsaechlich lesbaren Text ableiten, nicht aus Logos oder Bildern
- Wenn kein Name eindeutig lesbar ist: null"""

RULES_DATE = """ERSTELLDATUM-REGELN:
- Das Datum an dem das Dokument ERSTELLT/AUSGESTELLT wurde, NICHT das Scan-Datum
- Format: YYYY-MM-DD
- Bei Rechnungen: Rechnungsdatum
- Bei Briefen: Briefdatum oben rechts
- Bei Vertraegen: Vertragsdatum/Unterschriftsdatum
- Wenn mehrere Daten im Dokument: Das prominenteste/offizielle Dokumentdatum nehmen
- Wenn kein klares Datum erkennbar: null"""

RULES_TAGS = """TAG-REGELN:
- Tags beschreiben das THEMA/den ZWECK des Dokuments -- worum geht es inhaltlich?
- VERBOTEN: Firmennamen, Personennamen, Dokumenttyp-Woerter (Rechnung, Lieferschein...), AGB/Rechtstexte
- GUT: Branche (Bau, Energie, KFZ), Lebensbereich (Gesundheit, Haushalt, Kinder), Kostenart (Versicherung, Kredit)
- Bevorzuge Tags aus der verfuegbaren Liste
- Wenn KEIN passender Tag in der Liste existiert, darfst du EINEN neuen kurzen Tag vorschlagen (z.B. "Fotografie", "Haustier")
- Lieber weniger aber treffende Tags als viele ungenaue"""

RULES_DOCTYPE = """DOKUMENTTYP-REGELN:
- Nur aus der verfuegbaren Liste waehlen, nichts erfinden
- Wenn nichts passt: null setzen
- WICHTIG -- Rechnungserkennung:
  * Enthaelt das Dokument eine Rechnungsnummer (RE-..., RG-..., INV-..., R-...) oder Netto/Brutto/MwSt-Angaben? -> immer "Rechnung"
  * Eine Zahlungsbestaetigung MIT Rechnungsnummer ist trotzdem eine "Rechnung", NICHT eine "Bestaetigung"
  * "Bestaetigung" nur fuer Auftrags- oder Bestellbestaetigung OHNE Rechnungsnummer
- Prioritaet: Rechnungsnummer im Titel/Text schlaegt Woerter wie "bestätigt" oder "Zahlungseingang" im Freitext"""

RULES_CUSTOM_FIELDS = """CUSTOM-FIELDS-FORMAT-REGELN:
- Wenn ein Wert NICHT im Dokument zu finden ist: null setzen, NICHT raten
- Bei Betraegen: Punkt als Dezimaltrenner, keine Waehrungszeichen, kein Tausendertrennzeichen (z.B. 1499.99 statt 1.499,99 EUR)
- Bei IBAN: Ohne Leerzeichen, komplett (z.B. DE89370400440532013000)
- Bei Datum: Format YYYY-MM-DD
- Bei Rechnungsnummern: Exakt wie im Dokument"""


# --- OpenAI System Prompt ---

SYSTEM_PROMPT_OPENAI = """Du bist ein praeziser Dokumenten-Klassifizierer fuer ein Paperless-ngx Dokumentenmanagementsystem.

Deine Aufgabe: Analysiere den Dokumentinhalt und bestimme die passenden Metadaten.

ALLGEMEINE REGELN:
- Nutze die verfuegbaren Tools um bestehende Tags, Korrespondenten, Dokumenttypen und Speicherpfade nachzuschlagen
- Bevorzuge IMMER bestehende Eintraege statt neue zu erfinden
- Suche mit verschiedenen Begriffen wenn der erste Versuch keine Treffer liefert
- Wenn du dir bei einem Feld unsicher bist, setze es auf null
- WICHTIG: Rufe ALLE verfuegbaren Tools auf! Insbesondere get_storage_paths und get_custom_field_definitions MUESSEN aufgerufen werden wenn aktiviert.

{{ RULES_TITLE }}

{{ RULES_TAGS }}

{{ RULES_CORRESPONDENT }}

{{ RULES_DOCTYPE }}

{{ RULES_DATE }}

SPEICHERPFAD-REGELN:
- Lies die Personen-Profile sorgfaeltig und ordne dem richtigen Pfad zu
- Achte auf Privat vs. Geschaeftlich bei den Profilen
- du MUSST get_storage_paths aufrufen um die Profile zu sehen!

CUSTOM-FIELDS-REGELN:
- Rufe get_custom_field_definitions auf um die aktiven Felder mit Extraktions-Prompts abzurufen
- Fuer jedes Feld: Folge genau dem extraction_prompt und den Beispielwerten
{{ RULES_CUSTOM_FIELDS }}
- custom_fields ist ein Objekt mit Feldnamen als Keys: {"Rechnungsnummer": "RE-2024-0815", "Betrag": 49.99}

PFLICHT-ERGEBNIS-FORMAT -- Deine Antwort MUSS dieses JSON-Schema haben:
{
  "title": "...",
  "tags": ["...", "..."],
  "correspondent": "...",
  "document_type": "...",
  "created_date": "YYYY-MM-DD",
  "storage_path_id": <ID-Zahl oder null>,
  "storage_path_reason": "Kurze Begruendung",
  "custom_fields": {"Feldname": "Wert"}
}
ALLE Felder muessen vorhanden sein, auch wenn der Wert null ist!"""


# Readable per-field default rules (shown in the UI)
FIELD_DEFAULTS = {
    "title": RULES_TITLE,
    "tags": RULES_TAGS,
    "correspondent": RULES_CORRESPONDENT,
    "document_type": RULES_DOCTYPE,
    "date": RULES_DATE,
}


# --- Ollama Prompts (with SAME rules as OpenAI) ---

SYSTEM_PROMPT_OLLAMA_ANALYZE = """Du bist ein Dokumenten-Klassifizierer. Extrahiere Informationen als JSON.

WICHTIGSTE REGEL -- LIES DEN TEXT GENAU:
- Die ERSTEN 1-3 ZEILEN des Dokumentinhalts enthalten fast immer die Dokumentbezeichnung!
- "Ausdruck der elektronischen Lohnsteuerbescheinigung fuer 2015" = Titel ist "Lohnsteuerbescheinigung 2015"
- "Gehaltsabrechnung Oktober 2023" = Titel ist "Gehaltsabrechnung Oktober 2023"
- Uebernimm was IM TEXT STEHT, erfinde NICHTS dazu!
- Zahlen die als "Pers.-Nr.", "Personalnummer", "eTIN", "Steuer-Nr." markiert sind, sind KEINE Dokumentreferenzen!

Antworte als JSON:
{
  "title": "Kurzer Titel AUS DEM TEXT (nicht erfinden!)",
  "correspondent": "Absender/Aussteller",
  "created_date": "YYYY-MM-DD oder null",
  "summary": "2-3 Saetze Zusammenfassung",
  "language": "de/en/..."
}

{{ RULES_TITLE }}

{{ RULES_CORRESPONDENT }}

{{ RULES_DATE }}

Antworte NUR mit dem JSON, kein anderer Text."""


SYSTEM_PROMPT_OLLAMA_TAGS = """Waehle aus der folgenden Tag-Liste die passenden Tags fuer das beschriebene Dokument.
Waehle 2-5 Tags. Bevorzuge Tags aus dieser Liste:

{{ AVAILABLE_TAGS }}

Falls KEIN passender Tag in der Liste existiert, darfst du EINEN neuen kurzen Tag vorschlagen.

Dokument-Zusammenfassung: {{ SUMMARY }}

Antworte als JSON-Array mit den Tag-Namen:
["Tag1", "Tag2", "Tag3"]

Antworte NUR mit dem JSON-Array, kein anderer Text."""


SYSTEM_PROMPT_OLLAMA_DOCTYPE = """Bestimme den Dokumenttyp anhand des folgenden Dokuments.

EXTRAHIERTE METADATEN:
- Titel: {{ TITLE }}
- Korrespondent: {{ CORRESPONDENT }}
- KI-Zusammenfassung: {{ SUMMARY }}

DOKUMENTINHALT (Anfang):
{{ CONTENT_SNIPPET }}

ENTSCHEIDUNGSREGELN:
- Rechnungsnummer (RE-..., RG-..., INV-..., R-...) im Inhalt? -> 'Rechnung'
- Netto/Brutto/MwSt-Angaben im Inhalt? -> 'Rechnung'
- 'Zahlungseingang bestaetigt' + Rechnungsnummer? -> trotzdem 'Rechnung'
- 'Bestaetigung' NUR fuer Auftrags-/Bestellbestaetigung OHNE Rechnungsnummer
- Monatlicher Kontoauszug? -> 'Kontoauszug'
- Vertrag/Kuendigungsschreiben? -> 'Vertrag'

VERFUEGBARE TYPEN: {{ AVAILABLE_TYPES }}

Antworte als JSON: {"document_type": "Name"}"""


SYSTEM_PROMPT_OLLAMA_STORAGE_PATH = """Ordne dieses Dokument dem BESTEN verfuegbaren Speicherpfad zu.

ERKANNTE DOKUMENT-INFOS:
- Titel: {{ TITLE }}
- Korrespondent: {{ CORRESPONDENT }}
- Dokumenttyp: {{ DOCUMENT_TYPE }}
- Tags: {{ TAGS }}
- Zusammenfassung: {{ SUMMARY }}

DOKUMENTINHALT (Anfang):
{{ CONTENT_SNIPPET }}

VERFUEGBARE SPEICHERPFADE:
{{ PATH_PROFILES }}

ENTSCHEIDUNGS-REGELN:
- Lies die Kontext-Beschreibungen der Profile sorgfaeltig — sie definieren wem/welchem Bereich ein Pfad zugeordnet ist
- Vergleiche Korrespondent, Tags, Titel und Dokumentinhalt mit den Profil-Kontexten
- Waehle IMMER den best-passenden Pfad, auch wenn die Zuordnung nicht 100% eindeutig ist
- null NUR dann setzen, wenn absolut KEINE sinnvolle Zuordnung moeglich ist (z.B. komplett unbekannter Kontext)
- Bei mehreren passenden Pfaden: Waehle den spezifischsten (z.B. Kind-Profil vor Familien-Profil)

Antworte als JSON:
{{"path_id": <ID oder null>, "reason": "Kurze Begruendung mit Bezug zum Profil-Kontext"}}

Antworte NUR mit dem JSON, kein anderer Text."""


SYSTEM_PROMPT_OLLAMA_CUSTOM_FIELDS = """Extrahiere die folgenden Felder aus dem Dokumentinhalt.
Kopiere die Werte GENAU so wie sie im Dokument stehen.

{{ FIELD_DEFINITIONS }}

REGELN:
- Wenn ein Feld NICHT im Dokument vorkommt: null setzen
- Betraege: Als Zahl mit Dezimalstellen (z.B. 2359.77). Komma durch Punkt ersetzen.
- IBAN/Kontonummer: Komplett abschreiben, alle Ziffern und Buchstaben
- Rechnungsnummern: Exakt wie im Dokument

BEISPIEL:
{{"Rechnungsnummer": "RE-2024-0815", "Gesamtbetrag": 1499.99, "Kontonummer": "DE94300400000772924700", "Kundennummer": "10019"}}

Antworte NUR mit dem JSON, kein anderer Text."""


SYSTEM_PROMPT_OLLAMA_VERIFY = """Pruefe dieses Klassifizierungs-Ergebnis auf Vollstaendigkeit und Plausibilitaet.

DOKUMENT-ZUSAMMENFASSUNG: {{ SUMMARY }}

AKTUELLES ERGEBNIS:
- Titel: {{ TITLE }}
- Korrespondent: {{ CORRESPONDENT }}
- Dokumenttyp: {{ DOCUMENT_TYPE }}
- Tags: {{ TAGS }}
- Speicherpfad-ID: {{ STORAGE_PATH_ID }}
- Speicherpfad-Grund: {{ STORAGE_PATH_REASON }}
- Erstelldatum: {{ CREATED_DATE }}

VERFUEGBARE SPEICHERPFADE:
{{ STORAGE_PATHS }}

PRUEF-REGELN:
1. Wenn storage_path_id null ist aber Speicherpfade verfuegbar sind: Waehle den BESTEN Pfad
2. Tags muessen zum Dokumentinhalt passen, nicht zu generisch (Finanzen, Steuern, Umsatzsteuer = schlecht fuer eine Brunnenbohrung)
3. Alle Felder muessen befuellt sein wenn moeglich
4. Speicherpfad muss zur Person/zum Kontext passen

Antworte als JSON mit NUR den Feldern die du AENDERN willst.
Wenn alles korrekt ist, antworte mit leerem JSON: {}

Beispiel Korrektur: {"storage_path_id": 11, "storage_path_reason": "Privat Christian"}
Beispiel alles ok: {}

Antworte NUR mit dem JSON."""


PROMPTS = {
    "classifier_rules_title": RULES_TITLE,
    "classifier_rules_tags": RULES_TAGS,
    "classifier_rules_correspondent": RULES_CORRESPONDENT,
    "classifier_rules_doctype": RULES_DOCTYPE,
    "classifier_rules_date": RULES_DATE,
    "classifier_rules_custom_fields": RULES_CUSTOM_FIELDS,
    "classifier_ollama_analyze": SYSTEM_PROMPT_OLLAMA_ANALYZE,
    "classifier_ollama_doctype": SYSTEM_PROMPT_OLLAMA_DOCTYPE,
    "classifier_ollama_storage_path": SYSTEM_PROMPT_OLLAMA_STORAGE_PATH,
    "classifier_ollama_custom_fields": SYSTEM_PROMPT_OLLAMA_CUSTOM_FIELDS,
    "classifier_ollama_verify": SYSTEM_PROMPT_OLLAMA_VERIFY,
    "classifier_openai": SYSTEM_PROMPT_OPENAI,
}


# ── Local model recommendations (for UI) ─────────────────────────────────────

THINKING_MODEL_PREFIXES = ("qwen3", "deepseek-r1", "qwq")

LOCAL_RECOMMENDED_MODELS = {
    "qwen2.5:3b": {
        "text": "★ TOP-EMPFEHLUNG -- Schnell (~5-10s), praezises JSON, ideal fuer Klassifizierung",
        "category": "standard", "speed": "schnell", "quality": "gut",
    },
    "qwen2.5:7b": {
        "text": "★ BESTE QUALITAET -- Etwas langsamer, dafuer hoehere Trefferquote",
        "category": "standard", "speed": "mittel", "quality": "sehr gut",
    },
    "gemma2:2b": {
        "text": "Ultraschnell, kompakt -- Google-Modell, gut fuer einfache Dokumente",
        "category": "standard", "speed": "sehr schnell", "quality": "befriedigend",
    },
    "llama3.2:3b": {
        "text": "Schnell, kompakt -- gute Alternative zu qwen2.5:3b",
        "category": "standard", "speed": "schnell", "quality": "gut",
    },
    "phi3:mini": {
        "text": "Microsoft 3.8B -- stark bei strukturierten Aufgaben",
        "category": "standard", "speed": "schnell", "quality": "gut",
    },
    "llama3.1:8b": {
        "text": "Meta 8B -- solide, gute deutsche Sprachkenntnisse",
        "category": "standard", "speed": "mittel", "quality": "gut",
    },
    "gemma2:9b": {
        "text": "Google 9B -- praezise bei strukturiertem Output",
        "category": "standard", "speed": "mittel", "quality": "sehr gut",
    },
    "mistral:7b": {
        "text": "Mistral 7B -- gute europaeische Sprachunterstuetzung",
        "category": "standard", "speed": "mittel", "quality": "gut",
    },
    "qwen2.5:14b": {
        "text": "Premium-Qualitaet, braucht >10GB VRAM",
        "category": "standard", "speed": "langsam", "quality": "exzellent",
    },
    "qwen3:4b": {
        "text": "⚠ THINKING-Modell -- denkt nach (langsamer), aber JSON-Modus erzwungen",
        "category": "thinking", "speed": "langsam", "quality": "gut",
    },
    "qwen3:8b": {
        "text": "⚠ THINKING-Modell -- denkt nach (langsamer), aber JSON-Modus erzwungen",
        "category": "thinking", "speed": "langsam", "quality": "sehr gut",
    },
    "qwen3.5:9b": {
        "text": "⚠ THINKING-Modell -- denkt nach (deutlich langsamer, braucht mehr VRAM)",
        "category": "thinking", "speed": "sehr langsam", "quality": "sehr gut",
    },
    "deepseek-r1:8b": {
        "text": "⚠ THINKING-Modell -- Reasoning-fokussiert, langsam fuer Klassifizierung",
        "category": "thinking", "speed": "sehr langsam", "quality": "gut",
    },
}
