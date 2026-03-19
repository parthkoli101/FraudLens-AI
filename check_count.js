const mongoose = require('mongoose');
require('dotenv').config();
const Transaction = require('./models/Transaction');

async function checkCount() {
    await mongoose.connect(process.env.MONGO_URI || 'mongodb://localhost:27017/fraudlens');
    const count = await Transaction.countDocuments();
    console.log('TRANS_COUNT:' + count);
    process.exit(0);
}
checkCount();
