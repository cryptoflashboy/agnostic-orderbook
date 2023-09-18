//! The market state struct tracks metadata and security information about the agnostic orderbook system and its
//! relevant accounts.
pub use crate::state::orderbook::{OrderSummary, ORDER_SUMMARY_SIZE};
#[cfg(feature = "no-entrypoint")]
pub use crate::utils::get_spread;
use bytemuck::{Pod, Zeroable};
use solana_program::{entrypoint::ProgramResult, msg, program_error::ProgramError, pubkey::Pubkey};
use std::{convert::TryFrom, mem::size_of};

use super::{AccountTag, ACCOUNT_TAG_INDEX, ACCOUNT_TAG_LENGTH};

#[derive(Debug, Copy, Clone, Pod, Zeroable)]
#[repr(C)]
/// The orderbook market's central state
pub struct MarketState {
    /// The public key of the orderbook's event queue account
    pub event_queue: Pubkey,
    /// The public key of the orderbook's bids account
    pub bids: Pubkey,
    /// The public key of the orderbook's asks account
    pub asks: Pubkey,
    /// The minimum order size that can be inserted into the orderbook after matching.
    pub min_base_order_size: u64,
    /// Tick size (FP32)
    pub tick_size: u64,
}

impl MarketState {
    /// Expected size in bytes of MarketState
    pub const LEN: usize = size_of::<Self>();

    #[allow(missing_docs)]
    pub fn initialize(
        buffer: &mut [u8],
        expected_tag: AccountTag,
    ) -> Result<&mut Self, ProgramError> {
        match AccountTag::try_from(&buffer[ACCOUNT_TAG_INDEX..ACCOUNT_TAG_LENGTH]) {
            Ok(a) => {
                if a != expected_tag {
                    msg!("Invalid account tag for market!");
                    return Err(ProgramError::InvalidAccountData);
                }
                bytemuck::bytes_of(&(AccountTag::Market as u64))
                    .iter()
                    .enumerate()
                    .for_each(|(idx, byte)| buffer[ACCOUNT_TAG_INDEX + idx] = *byte);
            }
            Err(e) => {
                return Err(e);
            }
        };

        let (_, data) = buffer.split_at_mut(ACCOUNT_TAG_LENGTH);

        Ok(bytemuck::from_bytes_mut(data))
    }

    #[allow(missing_docs)]
    pub fn from_buffer(buffer: &[u8], expected_tag: AccountTag) -> Result<&Self, ProgramError> {
        match AccountTag::try_from(&buffer[ACCOUNT_TAG_INDEX..8]) {
            Ok(a) => {
                if a != expected_tag {
                    msg!("Invalid account tag for market!");
                    return Err(ProgramError::InvalidAccountData);
                }
            }
            Err(e) => {
                return Err(e);
            }
        };

        let (_, data) = buffer.split_at(ACCOUNT_TAG_LENGTH);

        Ok(bytemuck::from_bytes(data))
    }

    #[allow(missing_docs)]
    pub fn check_buffer_size(account_data: &[u8]) -> ProgramResult {
        if account_data.len() != 8 + MarketState::LEN {
            msg!("Invalid market size!");
            return Err(ProgramError::InvalidAccountData);
        }
        Ok(())
    }
}

#[test]
fn market_cast() {
    let mut buffer = [0u8; MarketState::LEN + 8];
    let r = MarketState::from_buffer(&mut buffer, AccountTag::Market);
    assert!(r.is_err());
    assert_eq!(r.unwrap_err(), ProgramError::InvalidAccountData)
}
