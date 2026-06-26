---
title: Projects
permalink: /projects/
---

{% for project in site.data.profile.projects %}
## {{ project.title }}

*{{ project.label }}*

{{ project.description }}

**Stack:** {{ project.stack | join: ", " }}

[{{ project.cta }}]({{ project.url }}){: .btn .btn--primary .btn--small }

{% endfor %}
