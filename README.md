# AI Agent Instruction Files

A small research site that looks at the markdown files developers write to instruct
AI coding agents — `CLAUDE.md`, `AGENTS.md`, and `SKILL.md` — and surfaces the
patterns that show up across hundreds of public GitHub repositories.

**👉 [analyze-claude-md.onrender.com](https://analyze-claude-md.onrender.com/)**

## What it does

Every day the site scrapes public examples of each file type from GitHub, runs
topic modeling (NMF) over the text, and renders the results as:

- **Topic cards** showing the dominant themes and the words that define them
- **Term trends** tracking how individual words rise and fall across daily runs
- **Topic evolution** measuring which topics stay stable and which churn over time

A tab nav at the top of the homepage switches between the three file types so you
can compare the conventions developers use for each.

## Why it exists

These instruction files are a young genre. There aren't established conventions
yet — every project invents its own structure. This site is an attempt to look
across the corpus and see what's actually catching on: what people put in their
`CLAUDE.md`, how `AGENTS.md` differs in practice, and how the new `SKILL.md`
format is shaping up as it spreads.

It's a curiosity project, not a product. Have a look around the
[live site](https://analyze-claude-md.onrender.com/) — that's the whole thing.
