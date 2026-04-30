CHAT_SYSTEM_PROMPT = (
    "Du bist ein hilfreicher Assistent der Fragen zu Dokumenten beantwortet. "
    "Antworte basierend auf dem bereitgestellten Kontext."
)

# Placeholders: {context}, {question}
CHAT_USER_CONTEXT_TEMPLATE = (
    "Kontext aus den Dokumenten:\n\n{context}\n\n---\n\nFrage: {question}\n\n"
    "Beantworte die Frage basierend auf dem Kontext. "
    "Zitiere die verwendeten Quellen mit ihrer Nummer aus dem Kontext: "
    "z.B. [3] für 'Quelle 3', [7] für 'Quelle 7'. "
    "Wenn du nach Fakten wie Geburtsdaten suchst, liste ALLE Fundstellen aus allen Quellen auf."
)

# Placeholders: {history_context}, {question}
QUERY_REWRITE_PROMPT = (
    "Du bist Suchexperte für ein deutsches Dokumentenarchiv (Paperless-ngx). "
    "Erweitere die Suchanfrage um Synonyme, offizielle Dokumentnamen und "
    "relevante deutsche Fachbegriffe (z.B. 'getauft' → 'Taufurkunde Taufe Taufschein', "
    "'geboren' → 'Geburtsurkunde Geburtsschein', 'Rechnung' → 'Rechnung Rechnungsnummer Betrag'). "
    "Antworte NUR mit der erweiterten Suchanfrage, max. 25 Wörter, kein Erklärungstext."
    "{history_context}\n\n"
    "Anfrage: {question}"
)

# Placeholders: {doc_info}, {chunk_text}
CHUNK_CONTEXT_PROMPT = (
    "{doc_info}\n\n"
    "Textabschnitt:\n{chunk_text}\n\n"
    "Schreibe 1-2 Sätze Kontext der erklärt:\n"
    "- Zu welchem Dokument/Person dieser Abschnitt gehört\n"
    "- Welche konkreten Fakten er enthält (Namen, Daten, Beträge, Kennzeichen)\n"
    "- Für welche Suchanfragen er relevant ist\n"
    "Nur der Kontext, keine Erklärungen, keine Einleitung wie 'Dieser Abschnitt...'."
)

PROMPTS = {
    "chat_system": CHAT_SYSTEM_PROMPT,
    "chat_user_context": CHAT_USER_CONTEXT_TEMPLATE,
    "query_rewrite": QUERY_REWRITE_PROMPT,
    "chunk_context": CHUNK_CONTEXT_PROMPT,
}
