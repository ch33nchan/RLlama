// src/components/HomepageFeatures.js
import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'Modular Reward Engineering',
    description: (
      <>
        Break down complex reward functions into reusable components, 
        making your reward systems easier to design, test, and maintain.
      </>
    ),
  },
  {
    title: 'Transparent Reward Calculations',
    description: (
      <>
        See exactly how each component contributes to the total reward,
        making it easier to debug and improve your reward functions.
      </>
    ),
  },
  {
    title: 'Seamless Integration',
    description: (
      <>
        RLlama works with popular RL frameworks like OpenAI Gym and 
        Stable Baselines3, making it easy to incorporate into your projects.
      </>
    ),
  },
];

function Feature({title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}