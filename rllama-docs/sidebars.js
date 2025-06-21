module.exports = {
  docs: [
    {
      type: 'category',
      label: 'Introduction',
      items: ['introduction/overview', 'introduction/features'],
    },
    {
      type: 'category',
      label: 'Installation and Setup',
      items: ['installation/requirements', 'installation/methods', 'installation/verification', 'installation/configuration'],
    },
    {
      type: 'category',
      label: 'Getting Started',
      items: ['getting-started/basic-usage', 'getting-started/hello-world', 'getting-started/first-reward-system', 'getting-started/common-patterns'],
    },
    {
      type: 'category',
      label: 'Core Concepts',
      items: ['concepts/reward-engineering'],
    },
    {
      type: 'category',
      label: 'RewardEngine',
      items: ['reward-engine/creating', 'reward-engine/adding-components', 'reward-engine/setting-weights', 'reward-engine/computing-rewards', 'reward-engine/analyzing-contributions'],
    },
    {
      type: 'category',
      label: 'Examples',
      items: ['examples/overview', 'examples/cartpole', 'examples/lunar-lander', 'examples/mountain-car', 'examples/maze-environment'],
    },
    'glossary',
    'cheatsheet',
  ],
};
