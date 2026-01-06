from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
import httpx
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from collections import defaultdict
import asyncio

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cost Settings
COST_FORWARD_SHIPPING = 80.0
COST_RTO_PENALTY = 29.0

# Store configurations (URLs only - tokens come from frontend)
STORE_CONFIG = {
    "wonder_care": {
        "name": "Wonder Care",
        "shop_url": "https://atppph-ha.myshopify.com"
    },
    "monzo": {
        "name": "Monozo",
        "shop_url": "https://icwi40-p0.myshopify.com"
    }
}

def get_store_with_token(store_id: str, token: str, shop_url: str) -> Dict:
    """Get store config with token from request"""
    if store_id not in STORE_CONFIG:
        return None
    return {
        "name": STORE_CONFIG[store_id]["name"],
        "shop_url": shop_url.rstrip("/") if shop_url else STORE_CONFIG[store_id]["shop_url"],
        "access_token": token
    }

def get_shopify_headers(store: Dict) -> Dict:
    return {
        "X-Shopify-Access-Token": store["access_token"],
        "Content-Type": "application/json"
    }

def get_shopify_url(store: Dict, endpoint: str) -> str:
    return f"{store['shop_url']}/admin/api/2024-01/{endpoint}"

def get_order_status(order: dict) -> str:
    """Priority-based order status detection"""
    tags = order.get('tags', '').lower()
    fin_status = order.get('financial_status', '').lower()
    ful_status = order.get('fulfillment_status') or ''
    is_cancelled = order.get('cancelled_at') is not None
    
    if 'rto' in tags or 'undelivered' in tags or 'return' in tags:
        return 'rto'
    if fin_status == 'paid':
        return 'delivered'
    if fin_status == 'voided' or ful_status == 'restocked' or is_cancelled:
        return 'cancelled'
    if fin_status == 'pending' and ful_status == 'fulfilled':
        return 'transit'
    return 'pending'

async def fetch_all_orders_with_retry(client: httpx.AsyncClient, store: Dict, params: Dict, max_retries: int = 3) -> List[Dict]:
    """Fetch all orders with pagination and retry logic"""
    all_orders = []
    url = get_shopify_url(store, "orders.json")
    headers = get_shopify_headers(store)
    current_params = params.copy()
    
    while url:
        retries = 0
        while retries < max_retries:
            try:
                response = await client.get(url, headers=headers, params=current_params if current_params else None)
                
                if response.status_code == 429:
                    retry_after = float(response.headers.get('Retry-After', 2))
                    await asyncio.sleep(retry_after)
                    retries += 1
                    continue
                
                response.raise_for_status()
                data = response.json()
                orders = data.get("orders", [])
                all_orders.extend(orders)
                
                link_header = response.headers.get('Link', '')
                url = None
                current_params = None
                
                if 'rel="next"' in link_header:
                    for link in link_header.split(','):
                        if 'rel="next"' in link:
                            url = link[link.find("<")+1:link.find(">")]
                            break
                
                if url:
                    await asyncio.sleep(0.5)
                break
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    retry_after = float(e.response.headers.get('Retry-After', 2))
                    await asyncio.sleep(retry_after)
                    retries += 1
                else:
                    raise
            except Exception as e:
                retries += 1
                if retries < max_retries:
                    await asyncio.sleep(1)
                else:
                    raise
    
    return all_orders

@app.get("/api/stores")
async def get_stores():
    return [{"store_id": sid, "name": s["name"]} for sid, s in STORE_CONFIG.items()]

@app.post("/api/sync/{store_id}")
async def sync_orders(store_id: str):
    if store_id not in STORE_CONFIG:
        raise HTTPException(status_code=404, detail="Store not found")
    return {"status": "success", "synced": 0}

@app.get("/api/summary/{store_id}")
async def get_summary(
    store_id: str,
    x_store_token: Optional[str] = Header(None),
    x_shop_url: Optional[str] = Header(None)
):
    if store_id not in STORE_CONFIG:
        raise HTTPException(status_code=404, detail="Store not found")
    
    if not x_store_token:
        raise HTTPException(status_code=401, detail="Access token required")
    
    store = get_store_with_token(store_id, x_store_token, x_shop_url)
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            since_date = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            params = {
                "status": "any",
                "created_at_min": since_date,
                "limit": 250,
                "fields": "id,tags,total_price,financial_status,fulfillment_status,cancelled_at"
            }
            
            orders = await fetch_all_orders_with_retry(client, store, params)
            
            stats = {"total": 0, "delivered": 0, "rto": 0, "transit": 0, "cancelled": 0, "pending": 0}
            
            for order in orders:
                status = get_order_status(order)
                stats["total"] += 1
                stats[status] += 1
            
            return {
                "total_orders": stats["total"],
                "cancelled": stats["cancelled"],
                "delivered": stats["delivered"],
                "in_transit": stats["transit"],
                "paid_orders": stats["delivered"],
                "pending_orders": stats["pending"]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/profit/{store_id}")
async def get_profit(
    store_id: str,
    x_store_token: Optional[str] = Header(None),
    x_shop_url: Optional[str] = Header(None)
):
    if store_id not in STORE_CONFIG:
        raise HTTPException(status_code=404, detail="Store not found")
    
    if not x_store_token:
        raise HTTPException(status_code=401, detail="Access token required")
    
    store = get_store_with_token(store_id, x_store_token, x_shop_url)
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            since_date = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            params = {
                "status": "any",
                "created_at_min": since_date,
                "limit": 250,
                "fields": "id,tags,total_price,financial_status,fulfillment_status,cancelled_at"
            }
            
            orders = await fetch_all_orders_with_retry(client, store, params)
            
            revenue_realized = 0.0
            shipping_cost = 0.0
            rto_penalty = 0.0
            
            for order in orders:
                status = get_order_status(order)
                order_val = float(order.get('total_price', 0))
                
                if status == 'delivered':
                    revenue_realized += order_val
                    shipping_cost += COST_FORWARD_SHIPPING
                elif status == 'rto':
                    shipping_cost += COST_FORWARD_SHIPPING
                    rto_penalty += COST_RTO_PENALTY
                elif status == 'transit':
                    shipping_cost += COST_FORWARD_SHIPPING
            
            total_cost = shipping_cost + rto_penalty
            profit = revenue_realized - total_cost
            
            return {
                "revenue": round(revenue_realized),
                "cost": round(total_cost),
                "profit": round(profit)
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cancelled/{store_id}")
async def get_cancelled(
    store_id: str,
    x_store_token: Optional[str] = Header(None),
    x_shop_url: Optional[str] = Header(None)
):
    if store_id not in STORE_CONFIG:
        raise HTTPException(status_code=404, detail="Store not found")
    
    if not x_store_token:
        raise HTTPException(status_code=401, detail="Access token required")
    
    store = get_store_with_token(store_id, x_store_token, x_shop_url)
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            since_date = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            params = {
                "status": "any",
                "created_at_min": since_date,
                "limit": 250,
                "fields": "id,order_number,tags,total_price,financial_status,fulfillment_status,cancelled_at,created_at"
            }
            
            orders = await fetch_all_orders_with_retry(client, store, params)
            
            cancelled = []
            for order in orders:
                status = get_order_status(order)
                if status in ['cancelled', 'rto']:
                    cancelled.append({
                        "order_id": f"#{order.get('order_number', order.get('id'))}",
                        "created_at": order.get("created_at"),
                        "total_price": float(order.get("total_price", 0)),
                        "reason": "RTO" if status == 'rto' else "Cancelled"
                    })
            
            return cancelled[:20]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sku-profit/{store_id}")
async def get_sku_profit(
    store_id: str,
    x_store_token: Optional[str] = Header(None),
    x_shop_url: Optional[str] = Header(None)
):
    if store_id not in STORE_CONFIG:
        raise HTTPException(status_code=404, detail="Store not found")
    
    if not x_store_token:
        raise HTTPException(status_code=401, detail="Access token required")
    
    store = get_store_with_token(store_id, x_store_token, x_shop_url)
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            since_date = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            params = {
                "status": "any",
                "created_at_min": since_date,
                "limit": 250
            }
            
            orders = await fetch_all_orders_with_retry(client, store, params)
            
            sku_data = defaultdict(lambda: {"total_qty": 0, "total_revenue": 0, "delivered_qty": 0})
            
            for order in orders:
                status = get_order_status(order)
                for item in order.get("line_items", []):
                    sku = item.get("sku") or item.get("title", "UNKNOWN")
                    qty = item.get("quantity", 1)
                    price = float(item.get("price", 0)) * qty
                    
                    sku_data[sku]["total_qty"] += qty
                    if status == 'delivered':
                        sku_data[sku]["total_revenue"] += price
                        sku_data[sku]["delivered_qty"] += qty
            
            result = []
            for sku, data in sku_data.items():
                revenue = data["total_revenue"]
                shipping = data["delivered_qty"] * COST_FORWARD_SHIPPING
                profit = revenue - shipping
                result.append({
                    "sku": sku,
                    "total_qty": data["total_qty"],
                    "total_revenue": round(revenue),
                    "total_cost": round(shipping),
                    "profit": round(profit)
                })
            
            return sorted(result, key=lambda x: x["total_revenue"], reverse=True)[:15]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/orders/{store_id}")
async def get_orders(
    store_id: str,
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
    x_store_token: Optional[str] = Header(None),
    x_shop_url: Optional[str] = Header(None)
):
    if store_id not in STORE_CONFIG:
        raise HTTPException(status_code=404, detail="Store not found")
    
    if not x_store_token:
        raise HTTPException(status_code=401, detail="Access token required")
    
    store = get_store_with_token(store_id, x_store_token, x_shop_url)
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            params = {"status": "any", "limit": 250}
            
            if from_date:
                params["created_at_min"] = f"{from_date}T00:00:00+00:00"
            if to_date:
                params["created_at_max"] = f"{to_date}T23:59:59+00:00"
            
            orders = await fetch_all_orders_with_retry(client, store, params)
            
            result = []
            for order in orders:
                status = get_order_status(order)
                fulfillments = order.get("fulfillments", [])
                tracking_id = None
                if fulfillments:
                    tracking_id = fulfillments[0].get("tracking_number")
                
                customer = order.get("customer", {})
                customer_name = None
                if customer:
                    first = customer.get("first_name", "")
                    last = customer.get("last_name", "")
                    customer_name = f"{first} {last}".strip() or None
                
                result.append({
                    "order_id": f"#{order.get('order_number', order.get('id'))}",
                    "created_at": order.get("created_at"),
                    "total_price": float(order.get("total_price", 0)),
                    "status": status,
                    "tracking_id": tracking_id,
                    "customer_name": customer_name,
                    "items_count": sum(item.get("quantity", 1) for item in order.get("line_items", []))
                })
            
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trends/{store_id}")
async def get_trends(
    store_id: str,
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
    x_store_token: Optional[str] = Header(None),
    x_shop_url: Optional[str] = Header(None)
):
    if store_id not in STORE_CONFIG:
        raise HTTPException(status_code=404, detail="Store not found")
    
    if not x_store_token:
        raise HTTPException(status_code=401, detail="Access token required")
    
    store = get_store_with_token(store_id, x_store_token, x_shop_url)
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            params = {"status": "any", "limit": 250}
            
            if from_date:
                params["created_at_min"] = f"{from_date}T00:00:00+00:00"
            if to_date:
                params["created_at_max"] = f"{to_date}T23:59:59+00:00"
            
            orders = await fetch_all_orders_with_retry(client, store, params)
            
            weekly_data = defaultdict(lambda: {"units": 0, "revenue": 0.0, "shipping": 0.0})
            
            for order in orders:
                status = get_order_status(order)
                if status != 'delivered':
                    continue
                
                created = order.get("created_at", "")
                if created:
                    dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                    week_start = dt - timedelta(days=dt.weekday())
                    week_key = week_start.strftime("%d %b")
                    
                    units = sum(item.get("quantity", 1) for item in order.get("line_items", []))
                    revenue = float(order.get("total_price", 0))
                    
                    weekly_data[week_key]["units"] += units
                    weekly_data[week_key]["revenue"] += revenue
                    weekly_data[week_key]["shipping"] += COST_FORWARD_SHIPPING
            
            result = []
            for period, data in sorted(weekly_data.items(), key=lambda x: datetime.strptime(x[0], "%d %b")):
                profit = data["revenue"] - data["shipping"]
                result.append({
                    "period": period,
                    "unitsSold": data["units"],
                    "salesRevenue": round(data["revenue"]),
                    "netProfit": round(profit)
                })
            
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cancellations-by-product/{store_id}")
async def get_cancellations_by_product(
    store_id: str,
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
    x_store_token: Optional[str] = Header(None),
    x_shop_url: Optional[str] = Header(None)
):
    if store_id not in STORE_CONFIG:
        raise HTTPException(status_code=404, detail="Store not found")
    
    if not x_store_token:
        raise HTTPException(status_code=401, detail="Access token required")
    
    store = get_store_with_token(store_id, x_store_token, x_shop_url)
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            params = {"status": "any", "limit": 250}
            
            if from_date:
                params["created_at_min"] = f"{from_date}T00:00:00+00:00"
            if to_date:
                params["created_at_max"] = f"{to_date}T23:59:59+00:00"
            
            orders = await fetch_all_orders_with_retry(client, store, params)
            
            product_data = defaultdict(lambda: {
                "productName": "",
                "totalOrders": 0,
                "cancelledOrders": 0,
                "rtoOrders": 0,
                "lostRevenue": 0.0
            })
            
            for order in orders:
                status = get_order_status(order)
                
                for item in order.get("line_items", []):
                    sku = item.get("sku") or "UNKNOWN"
                    product_name = item.get("title", "Unknown Product")
                    price = float(item.get("price", 0)) * item.get("quantity", 1)
                    
                    product_data[sku]["productName"] = product_name
                    product_data[sku]["totalOrders"] += 1
                    
                    if status == 'cancelled':
                        product_data[sku]["cancelledOrders"] += 1
                        product_data[sku]["lostRevenue"] += price
                    elif status == 'rto':
                        product_data[sku]["rtoOrders"] += 1
                        product_data[sku]["lostRevenue"] += price + COST_FORWARD_SHIPPING + COST_RTO_PENALTY
            
            result = []
            for sku, data in product_data.items():
                total = data["totalOrders"]
                cancelled = data["cancelledOrders"]
                rto = data["rtoOrders"]
                
                result.append({
                    "sku": sku,
                    "productName": data["productName"],
                    "totalOrders": total,
                    "cancelledOrders": cancelled,
                    "rtoOrders": rto,
                    "cancellationRate": round((cancelled / total * 100) if total > 0 else 0, 1),
                    "rtoRate": round((rto / total * 100) if total > 0 else 0, 1),
                    "lostRevenue": round(data["lostRevenue"])
                })
            
            result.sort(key=lambda x: x["cancelledOrders"] + x["rtoOrders"], reverse=True)
            return result[:20]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"status": "API Running", "stores": list(STORE_CONFIG.keys())}
