"""Task interpretation shared by ordinary-reading and source-graph models."""

REQUIREMENT_READING_RULES = (
    "The input is a temporal requirement whose satisfaction is being checked. "
    "Do not invent an advisory or non-binding reading solely because a requirement uses 'should'; "
    "preserve genuinely explicit optional/advisory wording rather than silently turning it into an obligation. "
    "'Eventually' alone requires occurrence but supplies no fixed, numeric or measurable deadline. "
    "Do not invent a deadline reading or report missing deadline information as a capability gap when "
    "the source contains no deadline. Unbounded eventuality is executable. "
    "Whether the current position counts is a separate possible distinction; preserve it when genuinely unresolved. "
    "Keep explicit condition direction, repeated scope such as 'whenever', participants, negation and actual time bounds. "
    "These task rules are not source quotes and do not constitute human approval."
)
