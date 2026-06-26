---
title: Contact
permalink: /contact/
---

I am open to conversations around LLM systems, workflow orchestration, evaluation, scientific AI, and research-oriented product development.

## Links

{% for item in site.data.profile.socials %}
- **{{ item.label }}:** [{{ item.url }}]({{ item.url }})
{% endfor %}
