---
title: Experience
permalink: /experience/
classes: wide
---

{{ site.data.profile.experience_intro }}

<div class="timeline-list">
{% for role in site.data.profile.experience_timeline %}
  <article class="timeline-item">
    <p class="section-kicker">{{ role.period }}</p>
    <h2>{{ role.title }}</h2>
    <p class="meta-line">{{ role.org }}{% if role.location %} • {{ role.location }}{% endif %}</p>
    {% if role.summary %}
    <p>{{ role.summary }}</p>
    {% endif %}
    {% if role.bullets %}
    <ul class="content-list">
      {% for bullet in role.bullets %}
      <li>{{ bullet }}</li>
      {% endfor %}
    </ul>
    {% endif %}
  </article>
{% endfor %}
</div>
