const { test, expect } = require("@playwright/test");

const pages = [
  { path: "/", id: "home" },
  { path: "/about/", id: "about" },
  { path: "/projects/", id: "projects" },
  { path: "/learning/", id: "learning" },
  { path: "/research/", id: "research" },
  { path: "/publications/", id: "publications" },
];

for (const { path, id } of pages) {
  test(`${id} page renders`, async ({ page }) => {
    const response = await page.goto(path, { waitUntil: "networkidle" });
    expect(response.status()).toBeLessThan(400);
    await expect(page).toHaveScreenshot(`${id}.png`);
  });
}
