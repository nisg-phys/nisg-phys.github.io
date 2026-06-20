---
title: Research
eyebrow: Research Focus
lead: Themes, methods, and problem spaces that shape how I approach AI systems.
permalink: /research/
---

<section class="panel">
  <div class="section-heading">
    <div>
      <p class="eyebrow">Themes</p>
      <h2>Areas I am actively interested in</h2>
    </div>
  </div>
  <div class="card-grid">
    {% for theme in site.data.profile.research_themes %}
      <article class="card">
        <h3>{{ theme.title }}</h3>
        <p>{{ theme.description }}</p>
      </article>
    {% endfor %}
  </div>
</section>

<section class="panel section-grid">
  <div class="card">
    <p class="eyebrow">Method</p>
    <h2>My working style</h2>
    <p>
      I prefer AI systems that can be decomposed into understandable parts:
      retrieval, planning, execution, reflection, and evaluation. This makes it
      easier to debug failures and improve behavior systematically.
    </p>
  </div>
  <div class="card">
    <p class="eyebrow">Application Lens</p>
    <h2>Where this matters</h2>
    <p>
      Enterprise automation, technical reasoning, and scientific discovery all
      benefit from workflows that are constraint-aware and measurable. Those are
      the environments where I want AI systems to be most useful.
    </p>
  </div>
</section>

<section class="panel panel-accent">
  <div class="section-heading">
    <div>
      <p class="eyebrow">Technical Toolkit</p>
      <h2>Disciplines that support the research</h2>
    </div>
  </div>
  <div class="capability-grid">
    {% for group in site.data.profile.skill_groups %}
      <article class="card">
        <h3>{{ group.title }}</h3>
        <div class="tag-list">
          {% for item in group.items %}
            <span>{{ item }}</span>
          {% endfor %}
        </div>
      </article>
    {% endfor %}
  </div>
</section>
