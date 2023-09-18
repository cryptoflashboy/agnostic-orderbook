#![cfg(feature = "benchmarking")]
use agnostic_orderbook::{
    instruction::{cancel_order, new_order},
    state::{
        critbit::Slab,
        event_queue::EventQueue,
        market_state::MarketState,
        orderbook::{CallbackInfo, OrderBookState},
        AccountTag, SelfTradeBehavior, Side,
    },
};
use bonfida_utils::{bench::get_env_arg, BorshSize};
use borsh::{BorshDeserialize, BorshSerialize};
use bytemuck::{Pod, Zeroable};
use solana_program::pubkey::Pubkey;
use solana_program_test::{processor, ProgramTest};
use solana_sdk::account::Account;
pub mod common;
use crate::common::utils::sign_send_instructions;

#[tokio::test]
async fn main() {
    let program_test = prepare().await;
    run(program_test).await;
}

pub struct Context {
    test_context: ProgramTest,
    test_order_id: u128,
    market: Pubkey,
    event_queue: Pubkey,
    bids: Pubkey,
    asks: Pubkey,
    register: Pubkey,
}

async fn run(ctx: Context) {
    let Context {
        test_context,
        test_order_id,
        market,
        event_queue,
        bids,
        asks,
        register,
        ..
    } = ctx;
    let mut ctx = test_context.start_with_context().await;
    let instruction = new_order(
        new_order::Accounts {
            market: &market,
            event_queue: &event_queue,
            bids: &bids,
            asks: &asks,
        },
        register,
        new_order::Params {
            max_base_qty: 10_000_000,
            max_quote_qty: 10_000_000,
            limit_price: 1010 << 32,
            side: Side::Bid,
            match_limit: 25,
            callback_info: C(Pubkey::new_unique().to_bytes()),
            post_only: false,
            post_allowed: true,
            self_trade_behavior: SelfTradeBehavior::DecrementTake,
            max_ts: u64::MAX,
        },
    );
    sign_send_instructions(&mut ctx, vec![instruction], vec![])
        .await
        .unwrap();
    let instruction = cancel_order(
        cancel_order::Accounts {
            market: &market,
            event_queue: &event_queue,
            bids: &bids,
            asks: &asks,
        },
        register,
        cancel_order::Params {
            order_id: test_order_id,
        },
    );
    sign_send_instructions(&mut ctx, vec![instruction], vec![])
        .await
        .unwrap()
}
#[derive(Clone, Copy, Pod, Zeroable, BorshDeserialize, BorshSerialize, PartialEq)]
#[repr(C)]
pub struct C([u8; 32]);

impl CallbackInfo for C {
    /// The callback identity object used to detect self trading
    type CallbackId = Self;

    /// Retrives a reference to the callback identity object from the parent object
    fn as_callback_id(&self) -> &Self::CallbackId {
        &self
    }
}

impl BorshSize for C {
    fn borsh_len(&self) -> usize {
        32
    }
}

async fn prepare() -> Context {
    let order_capacity = get_env_arg(0).unwrap_or(1_000);
    let market_key = Pubkey::new_unique();
    let event_queue_key = Pubkey::new_unique();
    let bids_key = Pubkey::new_unique();
    let asks_key = Pubkey::new_unique();
    let register_key = Pubkey::new_unique();
    // Initialize the event queue
    let mut event_queue_buffer = (0..EventQueue::<C>::compute_allocation_size(order_capacity))
        .map(|_| 0u8)
        .collect::<Vec<_>>();
    // Initialize the orderbook
    let mut asks_buffer = (0..Slab::<C>::compute_allocation_size(order_capacity))
        .map(|_| 0u8)
        .collect::<Vec<_>>();
    // Initialize the orderbook
    let mut bids_buffer = asks_buffer.clone();
    Slab::<C>::initialize(&mut asks_buffer, &mut bids_buffer).unwrap();
    // Initialize the market
    let mut market_state_buffer = (0..MarketState::LEN + 8).map(|_| 0u8).collect::<Vec<_>>();
    {
        let market_state =
            MarketState::initialize(&mut market_state_buffer, AccountTag::Uninitialized).unwrap();
        *market_state = MarketState {
            event_queue: event_queue_key,
            bids: bids_key,
            asks: asks_key,
            min_base_order_size: 1,
            tick_size: 1,
        }
    }
    let asks_slab = Slab::<C>::from_buffer(&mut asks_buffer, AccountTag::Asks).unwrap();
    let bids_slab = Slab::<C>::from_buffer(&mut bids_buffer, AccountTag::Bids).unwrap();
    let mut orderbook = OrderBookState::<C> {
        bids: bids_slab,
        asks: asks_slab,
    };
    let mut event_queue =
        EventQueue::from_buffer(&mut event_queue_buffer, AccountTag::Uninitialized).unwrap();
    let mut asks_order_ids = Vec::with_capacity(order_capacity);
    let mut bids_order_ids = Vec::with_capacity(order_capacity);
    // Input orders
    for i in 0..order_capacity as u64 {
        // println!("{}", orderbook.asks.header.bump_index);
        let o = orderbook
            .new_order(
                new_order::Params {
                    max_base_qty: 1_000_000,
                    max_quote_qty: 1_000_000,
                    limit_price: (i + 1) << 32,
                    side: Side::Bid,
                    match_limit: 10,
                    callback_info: C(Pubkey::new_unique().to_bytes()),
                    post_only: true,
                    post_allowed: true,
                    self_trade_behavior: SelfTradeBehavior::DecrementTake,
                    max_ts: u64::MAX,
                },
                &mut event_queue,
                1,
                0,
            )
            .unwrap();
        bids_order_ids.push(o.posted_order_id.unwrap());
        let o = orderbook
            .new_order(
                new_order::Params {
                    max_base_qty: 1_000_000,
                    max_quote_qty: 1_000_000,
                    limit_price: (i + 1 + (order_capacity as u64)) << 32,
                    side: Side::Ask,
                    match_limit: 10,
                    callback_info: C(Pubkey::new_unique().to_bytes()),
                    post_only: true,
                    post_allowed: true,
                    self_trade_behavior: SelfTradeBehavior::DecrementTake,
                    max_ts: u64::MAX,
                },
                &mut event_queue,
                1,
                0,
            )
            .unwrap();
        asks_order_ids.push(o.posted_order_id.unwrap());
        event_queue.pop_n(10);
        // println!("{}", i);
    }
    // We choose the order id with maximum depth
    let test_order_id = asks_order_ids[asks_order_ids.len() / 2];

    // We initialize the Solana testing environment
    let mut program_test = ProgramTest::new(
        "agnostic_orderbook",
        agnostic_orderbook::ID,
        processor!(agnostic_orderbook::entrypoint::process_instruction),
    );

    let lamports: u64 = 100_000_000_000;

    drop(orderbook);

    let accounts_to_add = vec![
        (market_key, market_state_buffer),
        (event_queue_key, event_queue_buffer),
        (bids_key, bids_buffer),
        (asks_key, asks_buffer),
        (register_key, vec![0; 42]),
    ];

    for (k, data) in accounts_to_add.into_iter() {
        program_test.add_account(
            k,
            Account {
                lamports,
                data,
                owner: agnostic_orderbook::ID,
                ..Account::default()
            },
        )
    }
    Context {
        test_context: program_test,
        test_order_id,
        market: market_key,
        event_queue: event_queue_key,
        bids: bids_key,
        asks: asks_key,
        register: register_key,
    }
}
