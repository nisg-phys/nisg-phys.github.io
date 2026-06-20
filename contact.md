---
title: Contact
eyebrow: Connect
lead: Reach out for research collaborations, engineering opportunities, or conversations around agentic AI systems.
permalink: /contact/
---

<section class="panel section-grid">
  <div class="card">
    <p class="eyebrow">Open To</p>
    <h2>Collaborations across research and applied AI.</h2>
    <p>
      I am especially interested in conversations around LLM systems, workflow
      orchestration, evaluation, scientific AI, and research-oriented product
      development.
    </p>
  </div>

  <div class="card">
    <p class="eyebrow">Links</p>
    <div class="stack-list">
      {% for item in site.data.profile.socials %}
        <div>
          <h3>{{ item.label }}</h3>
          <p><a class="text-link" href="{{ item.url }}">{{ item.url }}</a></p>
        </div>
      {% endfor %}
    </div>
  </div>
</section>
