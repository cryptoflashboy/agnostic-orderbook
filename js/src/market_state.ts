import { Connection, PublicKey } from "@solana/web3.js";
import { Schema, deserialize } from "borsh";
import { Slab } from "./slab";
import BN from "bn.js";

///////////////////////////////////////////////
////// Market State
///////////////////////////////////////////////

export enum AccountTag {
  Initialized = 0,
  Market = 1,
  EventQueue = 2,
  Bids = 3,
  Asks = 4,
}

export class MarketState {
  tag: AccountTag;
  callerAuthority: PublicKey;
  eventQueue: PublicKey;
  bids: PublicKey;
  asks: PublicKey;
  callBackInfoLen: BN;
  feeBudget: BN;
  initialLamports: BN;

  static schema: Schema = new Map([
    [
      MarketState,
      {
        kind: "struct",
        fields: [
          ["accountFlags", "u8"],
          ["callerAuthority", [32]],
          ["eventQueue", [32]],
          ["bids", [32]],
          ["asks", [32]],
          ["callBackInfoLen", "u64"],
          ["feeBudget", "u64"],
          ["initialLamports", "u64"],
        ],
      },
    ],
  ]);

  constructor(arg: {
    tag: number;
    callerAuthority: Uint8Array;
    eventQueue: Uint8Array;
    bids: Uint8Array;
    asks: Uint8Array;
    callBackInfoLen: BN;
    feeBudget: BN;
    initialLamports: BN;
  }) {
    this.tag = arg.tag as AccountTag;
    this.callerAuthority = new PublicKey(arg.callerAuthority);
    this.eventQueue = new PublicKey(arg.eventQueue);
    this.bids = new PublicKey(arg.bids);
    this.asks = new PublicKey(arg.asks);
    this.callBackInfoLen = arg.callBackInfoLen;
    this.feeBudget = arg.feeBudget;
    this.initialLamports = arg.initialLamports;
  }

  static async retrieve(connection: Connection, market: PublicKey) {
    const accountInfo = await connection.getAccountInfo(market);
    if (!accountInfo?.data) {
      throw new Error("Invalid account provided");
    }
    return deserialize(
      this.schema,
      MarketState,
      accountInfo.data
    ) as MarketState;
  }

  async loadBidsSlab(connection: Connection) {
    const bidsInfo = await connection.getAccountInfo(this.bids);
    if (!bidsInfo?.data) {
      throw new Error("Invalid bids account");
    }
    return deserialize(Slab.schema, Slab, bidsInfo.data) as Slab;
  }

  async loadAsksSlab(connection: Connection) {
    const asksInfo = await connection.getAccountInfo(this.asks);
    if (!asksInfo?.data) {
      throw new Error("Invalid asks account");
    }
    return deserialize(Slab.schema, Slab, asksInfo.data) as Slab;
  }
}