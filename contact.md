---
title: Contact
permalink: /contact/
classes: wide
---

I am open to conversations about retrieval systems, agent workflows, evaluation, and scientific AI applications.

## Links

<div class="list-grid">
{% for item in site.data.profile.socials %}
  <article class="list-card">
    <h3>{{ item.label }}</h3>
    <p><a href="{{ item.url }}">{{ item.url }}</a></p>
  </article>
{% endfor %}
</div>
