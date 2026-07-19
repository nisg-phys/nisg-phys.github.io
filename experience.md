---
title: Experience
permalink: /experience/
classes: wide
---

The work below spans production AI systems, scientific computation, and model evaluation.

<div class="timeline-list">
{% for role in site.data.profile.experience %}
  <article class="timeline-item">
    <p class="section-kicker">{{ role.period }}</p>
    <h2>{{ role.title }}</h2>
    <p class="meta-line">{{ role.org }}</p>
    <p>{{ role.summary }}</p>
  </article>
{% endfor %}
</div>

## Education

<div class="list-grid">
{% for item in site.data.profile.education %}
  <article class="list-card">
    <h3>{{ item.degree }}</h3>
    <p>{{ item.school }}</p>
  </article>
{% endfor %}
</div>
