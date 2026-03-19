const mongoose = require('mongoose');
const fs = require('fs');
const csv = require('csv-parser');
require('dotenv').config();

const Transaction = require('./models/Transaction');

const TRANSACTIONS_CSV = 'upi_transactions.csv';
const CUSTOMERS_CSV = 'upi_customers.csv';
const MERCHANTS_CSV = 'upi_merchants.csv';

function readCSV(filepath) {
    return new Promise((resolve, reject) => {
        const rows = [];
        fs.createReadStream(filepath)
            .pipe(csv())
            .on('data', d => rows.push(d))
            .on('end', () => resolve(rows))
            .on('error', reject);
    });
}

async function seedData() {
    try {
        await mongoose.connect(process.env.MONGO_URI || 'mongodb://localhost:27017/fraudlens');
        console.log('✅ Connected to MongoDB');

        // ── Load all 3 CSVs ──────────────────────────────────────────────────
        console.log('📂 Loading CSVs...');
        const [txns, custs, merchs] = await Promise.all([
            readCSV(TRANSACTIONS_CSV),
            readCSV(CUSTOMERS_CSV),
            readCSV(MERCHANTS_CSV),
        ]);
        console.log(`   Transactions : ${txns.length}`);
        console.log(`   Customers    : ${custs.length}`);
        console.log(`   Merchants    : ${merchs.length}`);

        // ── Build lookup maps ────────────────────────────────────────────────
        const custMap = {};
        custs.forEach(c => { custMap[c.customer_uuid] = c; });
        const merchMap = {};
        merchs.forEach(m => { merchMap[m.merchant_uuid] = m; });

        // ── Clear existing data ──────────────────────────────────────────────
        await Transaction.deleteMany({});
        console.log('🗑  Cleared existing transactions');

        // ── Build documents ──────────────────────────────────────────────────
        const docs = [];
        for (const t of txns) {
            const c = custMap[t.customer_uuid] || {};
            const m = merchMap[t.merchant_uuid] || {};

            const amount = parseFloat(t.transaction_amount) || 0;
            const lastAmt = parseFloat(c.last_transaction_amount) || 1000;
            const balance = parseFloat(c.account_balance) || 0;
            const age = parseInt(c.age) || 0;
            const txnCount = parseInt(c.total_transactions_count) || 0;
            const amtRatio = parseFloat((amount / (lastAmt + 0.001)).toFixed(2));

            // Fraud flag derivation (mirrors ml_model.py logic)
            const newDevice = (t.customer_device_id && c.registered_device_id)
                ? t.customer_device_id !== c.registered_device_id : false;
            const newIp = (t.customer_ip_address && c.registered_ip_address)
                ? t.customer_ip_address !== c.registered_ip_address : false;
            const diffCity = (t.transaction_location && c.home_branch)
                ? t.transaction_location !== c.home_branch : false;

            // Assign scenario-based fraud probability
            let prob = Math.random() * 15;
            let flags = [];

            const hasAnomaly = Math.random() < 0.15;
            if (hasAnomaly) {
                const type = Math.floor(Math.random() * 4);
                if (type === 0) {
                    prob = 85 + Math.random() * 14;
                    flags = ['High Amount Spike', 'Unregistered Device'];
                } else if (type === 1) {
                    prob = 65 + Math.random() * 15;
                    flags = ['Diff City', 'Unknown IP'];
                } else if (type === 2) {
                    prob = 45 + Math.random() * 15;
                    flags = ['High Amount Ratio'];
                } else {
                    prob = 25 + Math.random() * 15;
                    flags = ['New Merchant'];
                }
            }

            // Add signal-based flags on top of scenario flags
            if (newDevice && !flags.includes('Unregistered Device')) flags.push('Unregistered Device');
            if (newIp && !flags.includes('Unknown IP')) flags.push('Unknown IP');
            if (diffCity && !flags.includes('Diff City')) flags.push('Diff City');
            if (amtRatio > 3 && !flags.includes('High Amount Spike')) flags.push('High Amount Spike');

            let risk = 'LOW';
            if (prob >= 80) risk = 'CRITICAL';
            else if (prob >= 60) risk = 'HIGH';
            else if (prob >= 40) risk = 'MEDIUM';

            docs.push({
                // ── Transaction ──────────────────────────────────────────────
                transaction_uuid: t.transaction_uuid,
                transaction_amount: amount,
                transaction_timestamp: new Date(t.transaction_timestamp),
                transaction_location: t.transaction_location || '',
                customer_device_id: t.customer_device_id || '',
                customer_ip_address: t.customer_ip_address || '',

                // ── Customer ─────────────────────────────────────────────────
                customer_uuid: t.customer_uuid,
                customer_name: c.full_name || ('User ' + t.customer_uuid.slice(-4)),
                customer_bank: c.bank_name || 'Unknown Bank',
                customer_age: age,
                customer_balance: balance,
                customer_last_amt: lastAmt,
                amt_ratio: amtRatio,
                customer_reg_device: c.registered_device_id || '',
                customer_reg_ip: c.registered_ip_address || '',
                customer_txn_count: txnCount,
                customer_upi: c.upi_id || '',
                customer_branch: c.home_branch || '',

                // ── Merchant ─────────────────────────────────────────────────
                merchant_uuid: t.merchant_uuid,
                merchant_name: m.merchant_name || ('Merchant ' + t.merchant_uuid.slice(-4)),
                merchant_upi: m.merchant_upi_id || '',
                merchant_bank: m.merchant_bank_name || '',
                merchant_branch: m.merchant_bank_branch || '',
                merchant_since: m.merchant_account_open_date ? String(m.merchant_account_open_date).slice(0, 10) : '',

                // ── ML / Fraud ────────────────────────────────────────────────
                fraud_probability: parseFloat(prob.toFixed(2)),
                prediction: prob >= 50 ? 1 : 0,
                risk_level: risk,
                flags: flags,
                new_device: newDevice,
                new_ip: newIp,
                diff_city: diffCity,
                is_processed: true,
            });
        }

        // ── Insert in batches ───────────────────────────────────────────────
        console.log(`📑 Inserting ${docs.length} documents...`);
        const batchSize = 500;
        for (let i = 0; i < docs.length; i += batchSize) {
            await Transaction.insertMany(docs.slice(i, i + batchSize), { ordered: false });
            console.log(`   Inserted rows ${i + 1}–${Math.min(i + batchSize, docs.length)}`);
        }

        console.log('🚀 Seeding complete! All customer & merchant data is now in MongoDB.');
        process.exit(0);
    } catch (err) {
        console.error('❌ Seeding error:', err);
        process.exit(1);
    }
}

seedData();
