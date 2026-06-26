---
title: Research
permalink: /research/
---

## Research themes

{% for theme in site.data.profile.research_themes %}
### {{ theme.title }}

{{ theme.description }}

{% endfor %}
## Working style

I prefer AI systems that can be decomposed into understandable parts: retrieval, planning, execution, reflection, and evaluation. This makes it easier to debug failures and improve behavior systematically.

## Where this matters

Enterprise automation, technical reasoning, and scientific discovery all benefit from workflows that are constraint-aware and measurable. Those are the environments where I want AI systems to be most useful.

## Technical toolkit

{% for group in site.data.profile.skill_groups %}
### {{ group.title }}

{{ group.items | join: ", " }}

{% endfor %}
