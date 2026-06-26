---
title: Experience
permalink: /experience/
---

{% for role in site.data.profile.experience %}
## {{ role.title }}

**{{ role.org }}**  
{{ role.period }}

{{ role.summary }}

{% endfor %}
## Education

{% for item in site.data.profile.education %}
- **{{ item.degree }}**  
  {{ item.school }}
{% endfor %}
