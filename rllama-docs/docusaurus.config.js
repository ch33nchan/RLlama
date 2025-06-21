
/** @type {import('@docusaurus/types').DocusaurusConfig} */
module.exports = {
  title: 'RLlama Documentation',
  tagline: 'A composable reward engineering framework for reinforcement learning',
  url: 'https://yourdomain.com',
  baseUrl: '/',
  onBrokenLinks: 'warn', // Changed from 'throw' to 'warn' to make development easier
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: 'ch33nchan',
  projectName: 'RLlama',
  themeConfig: {
    navbar: {
      title: 'RLlama',
      logo: {
        alt: 'RLlama Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'doc',
          docId: 'introduction/overview',
          position: 'left',
          label: 'Docs',
        },
        {
          href: 'https://github.com/ch33nchan/RLlama',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Getting Started',
              to: '/docs/introduction/overview',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub Discussions',
              href: 'https://github.com/ch33nchan/RLlama/discussions',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} RLlama. Built with Docusaurus.`,
    },
  },
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl:
            'https://github.com/ch33nchan/RLlama/edit/main/website/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
  scripts: [
    {
      src: 'https://unpkg.com/mermaid/dist/mermaid.min.js',
      async: true,
    },
    {
      src: '/js/mermaid-init.js',
      async: true,
    },
  ],
};
