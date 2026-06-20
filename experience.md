---
title: Experience
eyebrow: Career Path
lead: A blend of production engineering, scientific research, and evaluation-oriented AI work.
permalink: /experience/
---

<section class="panel">
  <div class="timeline">
    {% for role in site.data.profile.experience %}
      <article class="timeline-item">
        <p class="meta">{{ role.period }}</p>
        <h2>{{ role.title }}</h2>
        <p class="timeline-org">{{ role.org }}</p>
        <p>{{ role.summary }}</p>
      </article>
    {% endfor %}
  </div>
</section>

<section class="panel">
  <div class="section-heading">
    <div>
      <p class="eyebrow">Education</p>
      <h2>Academic background</h2>
    </div>
  </div>
  <div class="card-grid compact-grid">
    {% for item in site.data.profile.education %}
      <article class="card">
        <h3>{{ item.degree }}</h3>
        <p>{{ item.school }}</p>
      </article>
    {% endfor %}
  </div>
</section>
