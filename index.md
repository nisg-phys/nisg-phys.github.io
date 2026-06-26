---
title: "Nishant Gupta"
layout: splash
permalink: /
author_profile: false
excerpt: "AI researcher and engineer focused on agentic systems, evaluation pipelines, and scientific machine intelligence."
header:
  overlay_color: "#111827"
  overlay_filter: 0.35
classes: wide
---

{{ site.data.profile.summary }}

[View Projects](/projects/){: .btn .btn--primary }
[Get in Touch](/contact/){: .btn .btn--inverse }

## Current focus

**{{ site.data.profile.focus.title }}**  
{{ site.data.profile.focus.description }}

## Research themes

{% for theme in site.data.profile.research_themes %}
- **{{ theme.title }}**: {{ theme.description }}
{% endfor %}

## Selected projects

{% for project in site.data.profile.projects %}
### [{{ project.title }}]({{ project.url }})

{{ project.description }}

**Stack:** {{ project.stack | join: ", " }}

{% endfor %}
