const Database = require('better-sqlite3');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');

const DB_PATH = path.join(__dirname, 'busts.db');
const UPLOADS_DIR = path.join(__dirname, '..', 'uploads', 'busts');

// Ensure uploads directory exists
if (!fs.existsSync(UPLOADS_DIR)) {
  fs.mkdirSync(UPLOADS_DIR, { recursive: true });
}

// Initialize database
const db = new Database(DB_PATH);

// Create busts table
db.exec(`
  CREATE TABLE IF NOT EXISTS busts (
    id TEXT PRIMARY KEY,
    image_path TEXT NOT NULL,
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  )
`);

// Lorem ipsum descriptions for variety
const loremDescriptions = [
  "A contemplative figure captured in marble, this bust represents the timeless pursuit of wisdom and inner peace. The subtle expression suggests deep philosophical thought.",
  "This classical sculpture embodies the ideals of human form and proportion. The smooth contours reflect masterful craftsmanship and artistic vision.",
  "A study in serenity, this bust captures a moment of perfect stillness. The balanced features speak to universal concepts of beauty and harmony.",
  "Rendered with exquisite attention to detail, this sculpture invites meditation on the nature of consciousness and being.",
  "The understated elegance of this form demonstrates the power of simplicity in artistic expression. Each curve serves a deliberate purpose.",
  "A testament to the enduring appeal of classical aesthetics, this bust bridges ancient traditions with contemporary sensibilities.",
  "This enigmatic figure holds secrets behind its composed exterior. The neutral expression invites projection and interpretation.",
  "Crafted with precision and care, this sculpture represents humanity's eternal quest to capture the essence of existence in physical form.",
  "The clean lines and pure form of this bust speak to minimalist ideals while honoring traditional sculptural techniques.",
  "A meditation on identity and presence, this figure exists in a space between the real and the ideal."
];

// Function to seed the database with initial busts
function seedDatabase(sourceImagePath, count = 100) {
  // Check if already seeded
  const existingCount = db.prepare('SELECT COUNT(*) as count FROM busts').get().count;
  if (existingCount >= count) {
    console.log(`Database already contains ${existingCount} busts. Skipping seed.`);
    return existingCount;
  }

  // Check if source image exists
  if (!fs.existsSync(sourceImagePath)) {
    console.error(`Source image not found: ${sourceImagePath}`);
    return 0;
  }

  const sourceBuffer = fs.readFileSync(sourceImagePath);
  const ext = path.extname(sourceImagePath);

  const insert = db.prepare(`
    INSERT INTO busts (id, image_path, description)
    VALUES (?, ?, ?)
  `);

  const insertMany = db.transaction((items) => {
    for (const item of items) {
      insert.run(item.id, item.imagePath, item.description);
    }
  });

  const items = [];
  for (let i = existingCount; i < count; i++) {
    const id = uuidv4();
    const filename = `bust_${id}${ext}`;
    const destPath = path.join(UPLOADS_DIR, filename);
    
    // Copy image to uploads folder
    fs.writeFileSync(destPath, sourceBuffer);
    
    // Get a random description
    const description = loremDescriptions[i % loremDescriptions.length];
    
    items.push({
      id,
      imagePath: `/uploads/busts/${filename}`,
      description
    });
  }

  insertMany(items);
  console.log(`Seeded ${items.length} busts into database.`);
  return count;
}

// Function to get database instance
function getDatabase() {
  return db;
}

// Function to close database
function closeDatabase() {
  db.close();
}

module.exports = {
  getDatabase,
  closeDatabase,
  seedDatabase,
  DB_PATH,
  UPLOADS_DIR
};

// If run directly, seed the database
if (require.main === module) {
  const sourceImage = path.join(__dirname, '..', 'assets', 'bust-default.png');
  
  // Check for command line argument for source image
  const customSource = process.argv[2];
  const imagePath = customSource || sourceImage;
  
  console.log(`Seeding database with image: ${imagePath}`);
  const count = seedDatabase(imagePath, 120);
  console.log(`Total busts in database: ${count}`);
  closeDatabase();
}
