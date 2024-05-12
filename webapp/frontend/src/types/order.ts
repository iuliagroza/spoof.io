export interface Order {
    order_id: string;
    type: string;
    size: number;
    price: number;
    status: string;
    time: string;
    reason: string;
    remaining_size: number;
    trade_id: string;
}