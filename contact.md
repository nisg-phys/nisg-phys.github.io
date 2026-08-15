---
title: Contact
permalink: /contact/
classes: wide
---

{{ site.data.profile.contact_intro }}

<div class="list-grid">
{% for item in site.data.profile.socials %}
  <article class="list-card">
    <h3>{{ item.label }}</h3>
    <p><a href="{{ item.url }}">{{ item.url }}</a></p>
  </article>
{% endfor %}
</div>
