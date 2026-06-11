// pm2 process config for the unattended exhibition run.
// Usage:  npm i -g pm2   then   npm run start:kiosk
//
// Supervises server.js: restarts it automatically if it crashes, AND restarts it
// if its memory crosses the cap (defence-in-depth alongside the browser tab's own
// idle self-reload). Use this for the show — NOT `npm run dev` (nodemon restarts
// on any file touch and offers no crash recovery).
module.exports = {
  apps: [
    {
      name: 'aema',
      script: 'server.js',
      exec_mode: 'fork',
      instances: 1,
      autorestart: true,
      max_memory_restart: '800M',
      // Don't let pm2 watch the project — the server writes clips/thumbnails/frames
      // at runtime and we never want those to trigger a restart.
      watch: false,
      env: {
        NODE_ENV: 'production',
        PORT: 3000
      }
    }
  ]
};
