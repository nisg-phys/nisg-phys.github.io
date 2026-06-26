---
title: About
permalink: /about/
---

I work on AI systems that need more than a good prompt. My interests are in building structured workflows around language models, especially when the task requires memory, tool use, retrieval, or domain-specific constraints.

My transition from theoretical physics to AI engineering shaped how I think: start from first principles, make the system observable, and evaluate it with enough rigor that we can trust outputs in real settings.

## At a glance

- **Location:** {{ site.data.profile.location }}
- **Current focus:** {{ site.data.profile.focus.description }}

## Research themes

{% for theme in site.data.profile.research_themes %}
### {{ theme.title }}

{{ theme.description }}

{% endfor %}
## Education

{% for item in site.data.profile.education %}
- **{{ item.degree }}**  
  {{ item.school }}
{% endfor %}
