use bonfida_utils::BorshSize;
use borsh::{BorshDeserialize, BorshSerialize};
use bytemuck::{CheckedBitPattern, NoUninit};
use num_derive::{FromPrimitive, ToPrimitive};
use solana_program::program_error::ProgramError;
use std::convert::TryFrom;

pub use crate::state::orderbook::{OrderSummary, ORDER_SUMMARY_SIZE};
#[cfg(feature = "no-entrypoint")]
pub use crate::utils::get_spread;

/// Describes the orderbook's underlying data structure, the [`Slab`].
pub mod critbit;
pub mod event_queue;
pub mod market_state;
pub mod orderbook;

/// The index of the [`AccountTag`] in the account buffers.
pub const ACCOUNT_TAG_INDEX: usize = 0;

/// The length of the [`AccountTag`] in the account buffers.
pub const ACCOUNT_TAG_LENGTH: usize = 8;

#[derive(Clone, Copy, PartialEq, Debug)]
#[allow(missing_docs)]
#[repr(u8)]
/// Warning: the account tags are bitshifted to allow for standard tag usage in the program using the aob.
pub enum AccountTag {
    Uninitialized,
    Market = 1 << 7,
    EventQueue,
    Bids,
    Asks,
    Disabled,
}

impl Default for AccountTag {
    fn default() -> Self {
        Self::Uninitialized
    }
}

impl TryFrom<&[u8]> for AccountTag {
    type Error = ProgramError;

    fn try_from(value: &[u8]) -> Result<Self, Self::Error> {
        match bytemuck::from_bytes::<u64>(value) {
            0 => Ok(Self::Uninitialized),
            128 => Ok(Self::Market),
            129 => Ok(Self::EventQueue),
            130 => Ok(Self::Bids),
            131 => Ok(Self::Asks),
            132 => Ok(Self::Disabled),
            _ => {
                return Err(ProgramError::InvalidAccountData);
            }
        }
    }
}

#[derive(
    BorshDeserialize,
    BorshSerialize,
    Clone,
    Copy,
    PartialEq,
    FromPrimitive,
    ToPrimitive,
    Debug,
    BorshSize,
    CheckedBitPattern,
    NoUninit,
)]
#[repr(u8)]
#[allow(missing_docs)]
pub enum Side {
    Bid,
    Ask,
}

impl Side {
    /// Helper function to get the opposite side.
    pub fn opposite(&self) -> Self {
        match self {
            Side::Bid => Side::Ask,
            Side::Ask => Side::Bid,
        }
    }
}

#[derive(BorshDeserialize, BorshSerialize, Clone, PartialEq, FromPrimitive, BorshSize)]
/// Describes what happens when two order with identical callback informations are matched together
pub enum SelfTradeBehavior {
    /// The orders are matched together
    DecrementTake,
    /// The order on the provide side is cancelled. Matching for the current order continues and essentially bypasses
    /// the self-provided order.
    CancelProvide,
    /// The entire transaction fails and the program returns an error.
    AbortTransaction,
}

/// This byte flag is set for order_ids with side Bid, and unset for side Ask
pub const ORDER_ID_SIDE_FLAG: u128 = 1 << 63;

/// This helper function deduces an order's side from its order_id
pub fn get_side_from_order_id(order_id: u128) -> Side {
    if ORDER_ID_SIDE_FLAG & order_id != 0 {
        Side::Bid
    } else {
        Side::Ask
    }
}
