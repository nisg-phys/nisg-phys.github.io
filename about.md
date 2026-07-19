---
title: About
permalink: /about/
classes: wide
---

I build AI systems that need more than a prompt: retrieval, tool use, memory, evaluation, and domain-specific constraints all have to work together.

My background in theoretical physics shaped how I work: start from first principles, make behavior observable, and evaluate systems tightly enough to trust them in practice.

<div class="detail-grid">
  <article class="detail-card">
    <p class="section-kicker">Current focus</p>
    <h2>{{ site.data.profile.focus.title }}</h2>
    <p>{{ site.data.profile.focus.description }}</p>
  </article>
  <article class="detail-card">
    <p class="section-kicker">Based in</p>
    <h2>{{ site.data.profile.location }}</h2>
    <p>Working across AI engineering, evaluation, and scientific machine intelligence.</p>
  </article>
</div>

## Research interests

<div class="card-grid">
{% for theme in site.data.profile.research_themes %}
  <article class="content-card">
    <h3>{{ theme.title }}</h3>
    <p>{{ theme.description }}</p>
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
