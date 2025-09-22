import "dotenv/config";
import { createClient } from "@supabase/supabase-js";
import { createDeepSeek } from "@ai-sdk/deepseek";
import { generateText, tool, stepCountIs } from "ai";
import Medusa from "@medusajs/medusa-js";
import ollama from "ollama";
import { z } from "zod";
import { ZodError } from "zod";
import pino from "pino";

// Logger configuration
const logger = pino({
  level: process.env.LOG_LEVEL || "info",
  timestamp: pino.stdTimeFunctions.isoTime,
});

// Configuration management
const config = {
  supabase: {
    url: process.env.SUPABASE_URL,
    anonKey: process.env.SUPABASE_ANON_KEY,
  },
  medusa: {
    baseUrl: process.env.MEDUSA_BACKEND_URL || "http://localhost:9000",
    apiKey: process.env.MEDUSA_API_KEY || "",
    maxRetries: 3,
  },
  ai: {
    provider: process.env.AI_PROVIDER || "ollama",
    model: process.env.AI_MODEL || "llama3.2:latest",
    deepseekModel: process.env.DEEPSEEK_MODEL || "deepseek-chat",
    deepseekApiKey: process.env.DEEPSEEK_API_KEY || "",
  },
  session: {
    timeout: parseInt(process.env.SESSION_TIMEOUT_MS) || 30 * 60 * 1000, // 30 minutes
    cleanupInterval:
      parseInt(process.env.SESSION_CLEANUP_INTERVAL_MS) || 5 * 60 * 1000, // 5 minutes
  },
};

// Validate required environment variables
const validateConfig = () => {
  const requiredEnvVars = ["SUPABASE_URL", "SUPABASE_ANON_KEY"];
  const missingVars = requiredEnvVars.filter(
    (varName) => !process.env[varName]
  );

  if (missingVars.length > 0) {
    throw new Error(
      `Missing required environment variables: ${missingVars.join(", ")}`
    );
  }
};

validateConfig();

// Initialize clients
const supabase = createClient(config.supabase.url, config.supabase.anonKey);
const medusa = new Medusa(config.medusa);
const deepseek = createDeepSeek({
  apiKey: config.ai.deepseekApiKey,
});

// System prompts
const systemPrompts = {
  support: `You are a helpful assistant for customer support and copilot for support staff. 
  Act professionally and empathetically, but also be proactive in suggesting solutions. 
  Be concise and clear in your responses. If you cannot help, suggest escalating to a human agent. 
  Use the tools available to you to get customer info, check product details, and create support tickets when needed. 
  Format JSON data in a user-friendly way when presenting to users.`,
  bargaining: `You are a sales assistant. Help customers negotiate prices considering:
  - Minimum order quantities
  - Vendor minimum prices
  - Current inventory levels
  - Customer purchase history
  - Note that prices are represented in their lowest format (e.g., 1000 = $10.00 or NGN 10.00 etc).
  - Be polite and professional. If an offer is too low, explain why and encourage a higher offer.
  - Use the tools available to you to check product pricing and create discounts when offers are accepted.
  - Format JSON data in a user-friendly way when presenting to users.
  - Always refer to prices in the currency the customer is using.
  - Never reveal internal pricing details or cost structures.
  - Always aim to close the sale while maintaining a positive customer experience.
  - provide concise responses directed to the customer when responding to their offers.`,
};

// Session management
class SessionManager {
  constructor() {
    this.sessions = new Map();
    this.channels = new Map();
  }

  getSession(sessionId) {
    return this.sessions.get(sessionId);
  }

  setSession(sessionId, data) {
    this.sessions.set(sessionId, {
      ...data,
      lastActivity: Date.now(),
    });
  }

  updateSessionActivity(sessionId) {
    const session = this.getSession(sessionId);
    if (session) {
      session.lastActivity = Date.now();
    }
  }

  cleanupInactiveSessions() {
    const now = Date.now();
    for (const [sessionId, session] of this.sessions.entries()) {
      if (now - session.lastActivity > config.session.timeout) {
        this.removeSession(sessionId);
        logger.info(`Cleaned up inactive session: ${sessionId}`);
      }
    }
  }

  removeSession(sessionId) {
    this.sessions.delete(sessionId);
    const channel = this.channels.get(sessionId);
    if (channel) {
      supabase.removeChannel(channel);
      this.channels.delete(sessionId);
    }
  }

  getChannel(sessionId) {
    return this.channels.get(sessionId);
  }

  setChannel(sessionId, channel) {
    this.channels.set(sessionId, channel);
  }
}

class HaggleManager {
  constructor() {
    this.haggleChannels = new Map();
    this.presenceChannel = null;
    this.sessionChannels = new Map();
  }

  // Initialize haggle presence channel
  initializeHagglePresence() {
    this.presenceChannel = supabase.channel("haggle:presence:watcher", {
      config: {
        presence: {
          key: "haggle_service",
        },
      },
    });

    this.presenceChannel
      .on("presence", { event: "sync" }, () => {
        const presenceState = this.presenceChannel.presenceState();
        logger.debug("Haggle presence sync:", presenceState);
      })
      .on("broadcast", { event: "message" }, async (payload) => {
        logger.debug("Haggle message event:", payload);
        const { sessionId } = payload.payload || {};

        if (sessionId && !this.sessionChannels.has(sessionId)) {
          await this.setupHaggleSessionChannel(sessionId);
        }
      })
      .subscribe((status) => {
        if (status === "SUBSCRIBED") {
          logger.info("Haggle presence channel subscribed");
          this.presenceChannel.track({
            online_at: new Date().toISOString(),
            service: "haggle_service",
          });
        }
      });

    return this.presenceChannel;
  }

  // Setup a haggle session channel
  async setupHaggleSessionChannel(sessionId) {
    if (this.sessionChannels.has(sessionId)) {
      return this.sessionChannels.get(sessionId);
    }

    const sessionChannel = supabase.channel(sessionId);

    sessionChannel
      .on("presence", { event: "sync" }, () => {
        const presenceState = sessionChannel.presenceState();
        logger.debug(`Haggle session ${sessionId} presence:`, presenceState);
      })
      .on("presence", { event: "join" }, (payload) => {
        logger.debug(`User joined haggle session ${sessionId}:`, payload);
      })
      .on("presence", { event: "leave" }, (payload) => {
        logger.debug(`User left haggle session ${sessionId}:`, payload);
      })
      .on("broadcast", { event: "message" }, async (payload) => {
        await this.handleHaggleMessage(sessionId, sessionChannel, payload);
      })
      .on("broadcast", { event: "bargaining_request" }, async (payload) => {
        await this.handleBargainingRequest(sessionId, sessionChannel, payload);
      })
      .subscribe(async (status) => {
        if (status === "SUBSCRIBED") {
          logger.info(`Subscribed to haggle session channel: ${sessionId}`);

          // Track presence
          sessionChannel.track({
            online_at: new Date().toISOString(),
            session: sessionId,
            service: "haggle_session",
          });

          // Send welcome message
          await sessionChannel.send({
            type: "broadcast",
            event: "message",
            payload: {
              timestamp: new Date().toISOString(),
              message:
                "Welcome to our negotiation! What price are you considering for this product?",
            },
          });
        }
      });

    this.sessionChannels.set(sessionId, sessionChannel);
    logger.debug(`Set up haggle session channel: ${sessionId}`);

    return sessionChannel;
  }

  // Handle incoming haggle messages
  async handleHaggleMessage(sessionId, sessionChannel, payload) {
    try {
      const {
        message,
        productId,
        variantId,
        region_id,
        cart_id,
        currency_code,
        context,
      } = payload.payload || {};
      logger.debug(`Haggle message for session ${sessionId}:`, {
        message,
        productId,
        variantId,
        region_id,
        cart_id,
        currency_code,
      });

      console.log("Received haggle message:", message);

      if (!message || !productId) {
        throw new AppError("Invalid message payload", "INVALID_PAYLOAD");
      }
      // Get product info for negotiation bounds
      // const productInfo = await tools.checkProductPricing.execute({
      //   productId,
      //   quantity: 1,
      //   region_id,
      //   cart_id,
      //   currency_code,
      // });

      // if (!productInfo) {
      //   throw new AppError("Product not found", "PRODUCT_NOT_FOUND");
      // }

      // Find the variant by variantId, or fallback to the first variant
      // let variant = null;
      // if (variantId) {
      //   variant = (productInfo.variants || []).find((v) => v.id === variantId);
      // }
      // if (!variant) {
      //   variant = (productInfo.variants || [])[0];
      // }
      // const minPrice = productInfo.price;
      // const moq = productInfo.moq || 1;
      // const available = productInfo.availableQuantity || 0;

      // Generate counter offer using the dedicated tool
      const negotiationResult = await tools.generateCounterOffer.execute({
        productId,
        customerOffer: this.extractOfferFromMessage(message),
        quantity: 1,
        region_id,
        cart_id,
        currency_code,
      });

      await sessionChannel.send({
        type: "broadcast",
        event: "counter_offer",
        payload: {
          timestamp: new Date().toISOString(),
          counterOffer: negotiationResult.counterOffer,
          minPrice: negotiationResult.minPrice,
          message: negotiationResult.message,
          inventory: negotiationResult.inventory,
          moq: negotiationResult?.moq || 1,
        },
      });
    } catch (error) {
      logger.error({ error, sessionId }, "Error handling haggle message");

      await sessionChannel.send({
        type: "broadcast",
        event: "counter_offer",
        payload: {
          timestamp: new Date().toISOString(),
          message:
            "Unable to process your offer at this time. Please try again.",
        },
      });
    }
  }

  // Handle formal bargaining requests
  async handleBargainingRequest(sessionId, sessionChannel, payload) {
    try {
      const {
        productId,
        variantId,
        cart_id,
        counterOffer,
        customerId,
        context,
        region_id,
        currency_code,
      } = payload.payload || {};

      // Get product info to validate the offer
      const productInfo = await tools.checkProductPricing.execute({
        productId,
        quantity: 1,
        region_id,
        cart_id,
        currency_code,
      });

      if (!productInfo) {
        throw new AppError("Product not found", "PRODUCT_NOT_FOUND");
      }

      // console.log("Product info for bargaining:", productInfo);
      const minPrice = Math.ceil(productInfo.price * 0.9); // Use the price from checkProductPricing;
      const minAcceptablePrice = Math.ceil(minPrice * 1); // 10% above minPrice
      const counterOfferRounded = Math.floor(counterOffer);
      // console.log("Counter offer rounded:", counterOfferRounded);
      // Determine if the offer is acceptable
      const accepted = counterOfferRounded >= minAcceptablePrice;

      // console.log("Bargaining request:", {
      //   productId,
      //   variantId,
      //   counterOfferRounded,
      //   minPrice,
      //   minAcceptablePrice,
      //   accepted,
      //   customerId,
      //   cart_id,
      //   region_id,
      //   currency_code,
      // });

      let responsePayload = {};

      if (accepted) {
        // Create discount for accepted offer
        const discountResult = await this.createHaggleDiscount(
          productId,
          variantId,
          minPrice,
          counterOffer,
          sessionId,
          customerId,
          region_id,
          currency_code,
          productInfo.price
        );

        console.log("Created discount:", discountResult);

        // Apply discount to cart if provided
        if (cart_id && discountResult.discountCode) {
          await tools.applyDiscountToCart.execute({
            cartId: cart_id,
            discountCode: discountResult.discountCode,
          });
        }

        responsePayload = {
          accepted: true,
          discountCode: discountResult.discountCode,
          discountAmount: discountResult.discountAmount,
          expiresAt: discountResult.expiresAt,
          message: `Offer accepted! Discount code ${discountResult.discountCode} has been applied to your cart.`,
        };
      } else {
        // Generate polite rejection message
        const aiResponse = await aiGenerator.generateResponse(
          `The customer made a counter offer of ${counterOffer} for product ${productId}, 
           but our minimum acceptable price is ${minAcceptablePrice}. 
           Generate a polite, professional message explaining why we cannot accept this offer 
           and encourage them to make a higher offer.`,
          [],
          sessionId,
          "bargaining"
        );

        responsePayload = {
          accepted: false,
          message:
            aiResponse ||
            "Unfortunately, we cannot accept this offer. Please consider a higher offer.",
        };
      }

      await sessionChannel.send({
        type: "broadcast",
        event: "bargaining_response",
        payload: responsePayload,
      });
    } catch (error) {
      logger.error({ error, sessionId }, "Error handling bargaining request");

      await sessionChannel.send({
        type: "broadcast",
        event: "bargaining_response",
        payload: {
          accepted: false,
          message: "Error processing your offer. Please try again.",
        },
      });
    }
  }

  // Create a discount for an accepted haggle offer
  async createHaggleDiscount(
    productId,
    variantId,
    minPrice,
    counterOffer,
    sessionId,
    customerId,
    region_id,
    currency_code, price
  ) {
    const discountAmount = price - counterOffer;
    const expiresAt = new Date(Date.now() + 4 * 60 * 60 * 1000).toISOString(); // 4 hours

    // Generate a unique discount code
    const discountCode = `HAGGLE-${Math.random()
      .toString(36)
      .substr(2, 6)
      .toUpperCase()}`;

    try {
      // Create the discount in Medusa
      const { discount } = await medusa.admin.discounts.create({
        code: discountCode,
        rule: {
          type: "fixed",
          value: discountAmount,
          allocation: "item",
          description: `Haggle discount for product ${productId}`,
          conditions: [
            {
              operator: "in",
              products: [productId],
            },
          ],
        },
        is_dynamic: !true,
        usage_limit: 1,
        starts_at: new Date().toISOString(),
        ends_at: expiresAt,
        regions: [region_id],
        metadata: {
          haggle: true,
          sessionId,
          productId,
          variantId: variantId || null,
          customerId: customerId || null,
        },
      });

      return {
        discountCode,
        discountAmount,
        expiresAt,
        discountId: discount.id,
      };
    } catch (error) {
      logger.error(
        { error, productId, sessionId },
        "Failed to create haggle discount"
      );
      throw new AppError(
        "Failed to create discount",
        "DISCOUNT_CREATION_ERROR",
        error
      );
    }
  }

  // Extract offer amount from message
  extractOfferFromMessage(message) {
    if (typeof message === "number") {
      return message;
    } else if (typeof message === "string") {
      const match = message.match(/(\d+(\.\d+)?)/);
      return match ? parseFloat(match[1]) : 0;
    }
    return 0;
  }

  // Clean up inactive haggle sessions
  cleanupInactiveSessions() {
    const now = Date.now();
    const inactiveTimeout = 30 * 60 * 1000; // 30 minutes

    for (const [sessionId, channel] of this.sessionChannels.entries()) {
      // Check if session is inactive (simplified - you might want to track activity)
      if (now - channel.joinedAt > inactiveTimeout) {
        this.removeHaggleSession(sessionId);
        logger.info(`Cleaned up inactive haggle session: ${sessionId}`);
      }
    }
  }

  // Remove a haggle session
  async removeHaggleSession(sessionId) {
    const channel = this.sessionChannels.get(sessionId);
    if (channel) {
      await supabase.removeChannel(channel);
      this.sessionChannels.delete(sessionId);
      logger.debug(`Removed haggle session channel: ${sessionId}`);
    }
  }

  // Get all active haggle sessions
  getActiveSessions() {
    return Array.from(this.sessionChannels.keys());
  }
}

// Initialize the haggle manager
const haggleManager = new HaggleManager();

const sessionManager = new SessionManager();

// Error handling utilities
class AppError extends Error {
  constructor(message, code, details = null) {
    super(message);
    this.code = code;
    this.details = details;
    this.name = this.constructor.name;
  }
}

const handleError = (error, context = "") => {
  if (error instanceof AppError) {
    logger.warn({ error, context }, `AppError in ${context}`);
    return error;
  } else if (error instanceof ZodError) {
    const validationError = new AppError(
      "Validation failed",
      "VALIDATION_ERROR",
      error.errors
    );
    logger.warn(
      { error: validationError, context },
      `Validation error in ${context}`
    );
    return validationError;
  } else {
    logger.error({ error, context }, `Unexpected error in ${context}`);
    return new AppError("Internal server error", "INTERNAL_ERROR");
  }
};

// Rate limiting (simple in-memory implementation)
class RateLimiter {
  constructor(maxRequests, timeWindow) {
    this.requests = new Map();
    this.maxRequests = maxRequests;
    this.timeWindow = timeWindow;
  }

  check(sessionId) {
    const now = Date.now();
    const windowStart = now - this.timeWindow;

    let sessionRequests = this.requests.get(sessionId) || [];
    sessionRequests = sessionRequests.filter((time) => time > windowStart);

    if (sessionRequests.length >= this.maxRequests) {
      return false;
    }

    sessionRequests.push(now);
    this.requests.set(sessionId, sessionRequests);
    return true;
  }
}

const rateLimiter = new RateLimiter(
  parseInt(process.env.RATE_LIMIT_MAX_REQUESTS) || 100,
  parseInt(process.env.RATE_LIMIT_TIME_WINDOW_MS) || 15 * 60 * 1000 // 15 minutes
);

// Validation schemas
const schemas = {
  customerId: z.object({
    customerId: z.string().uuid("Invalid customer ID format"),
  }),
  productId: z.object({
    productId: z.string().uuid("Invalid product ID format"),
  }),
  orderId: z.object({
    orderId: z.string().uuid("Invalid order ID format"),
  }),
  sessionId: z.object({
    sessionId: z.string().min(1, "Session ID is required"),
  }),
};

function formatPrice({ amount, currencyCode, regionId }) {
  let localeCurrency = currencyCode

  // if (regionId) {
  //   const { region } = await medusa.regions.retrieve(regionId)
  //   localeCurrency = region.currency_code
  // }

  if (!localeCurrency) {
    throw new Error("Either currencyCode or regionId must be provided")
  }

  const formatter = new Intl.NumberFormat(undefined, {
    style: "currency",
    currency: localeCurrency.toUpperCase(),
    currencyDisplay: 'narrowSymbol'
  })

  return formatter.format(amount / 100) // convert from minor units
}


// Tool definitions with comprehensive error handling and validation
const tools = {
  generateCounterOffer: tool({
    description: "Generate a counter offer during price negotiation",
    inputSchema: z.object({
      productId: z
        .string()
        .min(1, "Invalid product ID format")
        .describe("The ID of the product being negotiated"),
      customerOffer: z
        .number()
        .min(0, "Offer must be a positive number")
        .describe("The customer's offer price"),
      customerId: z
        .string()
        .min(1, "Invalid customer ID format")
        .optional()
        .describe("ID of the customer (if available)"),
      quantity: z
        .number()
        .min(1, "Quantity must be at least 1")
        .default(1)
        .describe("Quantity of products being negotiated"),
      region_id: z
        .string()
        .optional()
        .describe("Region ID for region-specific pricing (optional)"),
      currency_code: z
        .string()
        .optional()
        .describe("Currency code for currency-specific pricing (optional)"),
      cart_id: z
        .string()
        .optional()
        .describe("Cart ID to get cart-specific pricing (optional)"),
    }),
    execute: async ({
      productId,
      customerOffer,
      customerId,
      quantity,
      region_id,
      currency_code,
      cart_id,
    }) => {
      try {
        logger.debug(
          { productId, customerOffer, region_id, currency_code, cart_id },
          "Generating counter offer"
        );

        // Get product info with all optional parameters
        const productInfo = await tools.checkProductPricing.execute({
          productId,
          quantity,
          region_id,
          currency_code,
          cart_id,
        });

        if (!productInfo) {
          throw new AppError("Product not found", "PRODUCT_NOT_FOUND");
        }

        const minPrice = Math.ceil(productInfo.price * 0.9); // Use the price from checkProductPricing
        const available = productInfo.availableQuantity;

        // Calculate negotiation parameters
        const minAcceptablePrice = Math.ceil(minPrice * 1); // 90% of min price
        const targetPrice = Math.ceil(minPrice * 1); // 95% of min price

        // Determine counter offer based on customer offer
        let counterOffer;
        let message;

        if (customerOffer >= minPrice) {
          // Customer offer meets or exceeds our price
          counterOffer = minPrice;
          message = `We accept your offer of ${formatPrice({amount: customerOffer, currencyCode: currency_code, regionId: region_id})}. The product is yours!`;
        } else if (customerOffer >= targetPrice) {
          // Customer offer is reasonable but below our price
          counterOffer = Math.min(
            customerOffer + Math.ceil((minPrice - customerOffer) * 0.3),
            minPrice
          );
          message = `We appreciate your offer. Our best counter offer is ${ formatPrice({amount: counterOffer, currencyCode: currency_code, regionId: region_id})}.`;
        } else if (customerOffer >= minAcceptablePrice) {
          // Customer offer is low but acceptable
          counterOffer = targetPrice;
          message = `We can offer you a special price of ${formatPrice({amount: counterOffer, currencyCode: currency_code, regionId: region_id})} (regular price: ${formatPrice({amount: minPrice, currencyCode: currency_code, regionId: region_id})}).`;
        } else {
          // Customer offer is too low
          counterOffer = minAcceptablePrice;
          message = `Unfortunately, we cannot accept offers below ${formatPrice({amount: minAcceptablePrice, currencyCode: currency_code, regionId: region_id})}. Our best price is ${formatPrice({amount: minPrice, currencyCode: currency_code, regionId: region_id})}.`;
        }

        // Check inventory
        if (available < quantity) {
          message += ` Note: Only ${available} items are currently available.`;
        }

        logger.debug(
          { productId, customerOffer, counterOffer },
          "Counter offer generated"
        );
        return {
          counterOffer,
          minPrice,
          minAcceptablePrice,
          message,
          inventory: available,
          region_id: productInfo.region_id,
          currency_code: productInfo.currency_code,
        };
      } catch (error) {
        throw new AppError(
          "Failed to generate counter offer",
          "COUNTER_OFFER_ERROR",
          error
        );
      }
    },
  }),

  getProductInfo: tool({
    description:
      "Retrieve comprehensive information about a product, including variants, prices, inventory, metadata, and store_id from Supabase",
    inputSchema: z.object({
      productId: z
        .string()
        .min(1, "Invalid product ID format")
        .describe("The ID of the product to retrieve"),
    }),
    execute: async ({ productId }) => {
      try {
        logger.debug({ productId }, "Fetching comprehensive product info");

        // Fetch product from Medusa
        const { product } = await medusa.admin.products.retrieve(productId, {
          expand: "store,variants,options,images",
        });

        if (!product) {
          throw new AppError("Product not found", "PRODUCT_NOT_FOUND");
        }

        // Fetch store_id from Supabase product table
        const { data: supabaseProduct, error: supabaseError } = await supabase
          .from("product")
          .select("store_id")
          .eq("id", productId)
          .single();

        if (supabaseError) {
          logger.warn(
            { productId, supabaseError },
            "Failed to fetch store_id from Supabase"
          );
        }

        return {
          id: product.id,
          title: product.title,
          description: product.description,
          status: product.status,
          thumbnail: product.thumbnail,
          handle: product.handle,
          type: product.type,
          collection: product.collection,
          tags: product.tags,
          createdAt: product.created_at,
          updatedAt: product.updated_at,
          variants: product.variants.map((variant) => ({
            id: variant.id,
            title: variant.title,
            sku: variant.sku,
            prices: variant.prices,
            inventory: variant.inventory_quantity,
            allowBackorder: variant.allow_backorder,
            manageInventory: variant.manage_inventory,
            minQuantity: variant.min_quantity,
            maxQuantity: variant.max_quantity,
            options: variant.options,
            createdAt: variant.created_at,
            updatedAt: variant.updated_at,
          })),
          options: product.options,
          images: product.images,
          metadata: product.metadata,
          storeId: supabaseProduct?.store_id || null,
          store: product?.store,
        };
      } catch (error) {
        if (error.status === 404) {
          throw new AppError("Product not found", "PRODUCT_NOT_FOUND", error);
        }
        throw new AppError(
          "Failed to fetch comprehensive product info",
          "PRODUCT_INFO_ERROR",
          error
        );
      }
    },
  }),
  getCustomerInfo: tool({
    description: "Get customer information from Medusa",
    inputSchema: z.object({
      customerId: z
        .string()
        .min(1, "Invalid customer ID format")
        .describe("The ID of the customer to retrieve"),
    }),
    execute: async ({ customerId }) => {
      try {
        logger.debug({ customerId }, "Fetching customer info");
        const { customer } = await medusa.admin.customers.retrieve(customerId);

        if (!customer) {
          throw new AppError("Customer not found", "CUSTOMER_NOT_FOUND");
        }

        logger.debug({ customerId }, "Successfully fetched customer info");
        return {
          id: customer.id,
          email: customer.email,
          firstName: customer.first_name,
          lastName: customer.last_name,
          phone: customer.phone,
          hasAccount: customer.has_account,
          orders: customer.orders,
          createdAt: customer.created_at,
          updatedAt: customer.updated_at,
        };
      } catch (error) {
        if (error.status === 404) {
          throw new AppError("Customer not found", "CUSTOMER_NOT_FOUND", error);
        }
        throw new AppError(
          "Failed to fetch customer info",
          "CUSTOMER_FETCH_ERROR",
          error
        );
      }
    },
  }),

  checkProductPricing: tool({
    description:
      "Check product pricing and inventory information with optional region, currency, and cart context",
    inputSchema: z.object({
      productId: z
        .string()
        .min(1, "Invalid product ID format")
        .describe("The ID of the product"),
      quantity: z
        .number()
        .min(1, "Quantity must be at least 1")
        .describe("Requested quantity"),
      regionId: z
        .string()
        .optional()
        .describe("Region ID for region-specific pricing (optional)"),
      currencyCode: z
        .string()
        .optional()
        .describe("Currency code for currency-specific pricing (optional)"),
      cartId: z
        .string()
        .optional()
        .describe("Cart ID to get cart-specific pricing (optional)"),
    }),
    execute: async ({
      productId,
      quantity,
      region_id,
      currency_code,
      cart_id,
    }) => {
      try {
        logger.debug(
          { productId, quantity, region_id, currency_code, cart_id },
          "Fetching product info"
        );

        // Fetch product with expanded variants and prices
        const { product } = await medusa.admin.products.retrieve(productId, {
          expand: "variants,variants.prices",
        });

        if (!product) {
          throw new AppError("Product not found", "PRODUCT_NOT_FOUND");
        }

        if (!product.variants || product.variants.length === 0) {
          throw new AppError("Product has no variants", "PRODUCT_NO_VARIANTS");
        }

        const variant = product.variants[0];

        // Get the appropriate price based on region/currency/cart context
        let price = 0;
        let selectedPrice = null;

        // Try to find the most specific price match
        if (variant.prices && variant.prices.length > 0) {
          // Priority 1: Exact match for region + currency
          if (region_id && currency_code) {
            selectedPrice = variant.prices.find(
              (p) =>
                p.region_id === region_id && p.currency_code === currency_code
            );
          }

          // Priority 2: Region match only
          if (!selectedPrice && region_id) {
            selectedPrice = variant.prices.find(
              (p) => p.region_id === region_id
            );
          }

          // Priority 3: Currency match only (including prices with null region_id)
          if (!selectedPrice && currency_code) {
            selectedPrice = variant.prices.find(
              (p) => p.currency_code === currency_code
            );
          }

          // Priority 4: Cart-specific prices (typically have null region_id)
          if (!selectedPrice && cart_id) {
            selectedPrice = variant.prices.find((p) => p.region_id === null);
          }

          // Priority 5: Fall back to the first available price
          if (!selectedPrice) {
            selectedPrice = variant.prices[0];
          }
        }

        if (selectedPrice) {
          price = selectedPrice.amount;
        }

        const minQuantity = variant.min_quantity || 1;
        const inventory = variant.inventory_quantity || 0;

        // Calculate extended price
        const extendedPrice = price * Math.max(quantity, minQuantity);

        // Check if requested quantity is available
        const available = inventory >= quantity;

        logger.debug({ productId }, "Successfully fetched product info");
        return {
          productId: product.id,
          productName: product.title,
          description: product.description,
          price,
          minQuantity,
          availableQuantity: inventory,
          requestedQuantity: quantity,
          extendedPrice,
          isAvailable: available,
          region_id: selectedPrice?.region_id || null,
          currency_code: selectedPrice?.currency_code || null,
          variants: product.variants.map((v) => ({
            id: v.id,
            title: v.title,
            prices: v.prices,
            inventory: v.inventory_quantity,
          })),
        };
      } catch (error) {
        if (error.status === 404) {
          throw new AppError("Product not found", "PRODUCT_NOT_FOUND", error);
        }
        throw new AppError(
          "Failed to fetch product info",
          "PRODUCT_FETCH_ERROR",
          error
        );
      }
    },
  }),

  createSupportTicket: tool({
    description: "Create a support ticket for human assistance",
    inputSchema: z.object({
      sessionId: z
        .string()
        .min(1, "Session ID is required")
        .describe("The session ID for the support ticket"),
      issue: z
        .string()
        .min(10, "Issue description must be at least 10 characters")
        .describe("Description of the issue"),
      priority: z
        .enum(["low", "medium", "high"])
        .default("medium")
        .describe("Priority level of the ticket"),
      category: z
        .string()
        .optional()
        .describe("Category of the issue (e.g., billing, technical, sales)"),
    }),
    execute: async ({ sessionId, issue, priority, category }) => {
      try {
        logger.debug({ sessionId }, "Creating support ticket");

        // Get customer info from session if available
        const session = sessionManager.getSession(sessionId);
        const customerId = session?.customer?.id || null;

        const { data, error } = await supabase
          .from("support_tickets")
          .insert([
            {
              session_id: sessionId,
              customer_id: customerId,
              issue,
              priority,
              category: category || "general",
              status: "open",
              created_at: new Date().toISOString(),
              updated_at: new Date().toISOString(),
            },
          ])
          .select();

        if (error) {
          throw new AppError(
            "Database error creating ticket",
            "TICKET_DB_ERROR",
            error
          );
        }

        logger.debug(
          { sessionId, ticketId: data[0].id },
          "Support ticket created successfully"
        );
        return data[0];
      } catch (error) {
        throw new AppError(
          "Failed to create support ticket",
          "TICKET_CREATION_ERROR",
          error
        );
      }
    },
  }),

  getOrderDetails: tool({
    description: "Retrieve details of a specific order",
    inputSchema: z.object({
      orderId: z
        .string()
        .min(1, "Invalid order ID format")
        .describe("The ID of the order"),
    }),
    execute: async ({ orderId }) => {
      try {
        logger.debug({ orderId }, "Fetching order details");
        const { order } = await medusa.admin.orders.retrieve(orderId);

        if (!order) {
          throw new AppError("Order not found", "ORDER_NOT_FOUND");
        }

        logger.debug({ orderId }, "Successfully fetched order details");
        return {
          id: order.id,
          status: order.status,
          customerId: order.customer_id,
          email: order.email,
          items: order.items,
          total: order.total,
          shippingAddress: order.shipping_address,
          billingAddress: order.billing_address,
          payments: order.payments,
          fulfillments: order.fulfillments,
          returns: order.returns,
          createdAt: order.created_at,
          updatedAt: order.updated_at,
        };
      } catch (error) {
        if (error.status === 404) {
          throw new AppError("Order not found", "ORDER_NOT_FOUND", error);
        }
        throw new AppError(
          "Failed to fetch order details",
          "ORDER_FETCH_ERROR",
          error
        );
      }
    },
  }),

  listCustomerOrders: tool({
    description: "List all orders for a specific customer",
    inputSchema: z.object({
      customerId: z
        .string()
        .min(1, "Invalid customer ID format")
        .describe("The ID of the customer"),
      limit: z
        .number()
        .min(1)
        .max(100)
        .default(10)
        .describe("Maximum number of orders to return"),
      offset: z
        .number()
        .min(0)
        .default(0)
        .describe("Number of orders to skip for pagination"),
    }),
    execute: async ({ customerId, limit, offset }) => {
      try {
        logger.debug({ customerId }, "Listing customer orders");
        const { orders, count } = await medusa.admin.orders.list({
          customer_id: customerId,
          limit,
          offset,
        });

        logger.debug(
          { customerId, count },
          "Successfully listed customer orders"
        );
        return {
          orders: orders.map((order) => ({
            id: order.id,
            status: order.status,
            total: order.total,
            createdAt: order.created_at,
            items: order.items.length,
          })),
          totalCount: count,
          hasMore: offset + orders.length < count,
        };
      } catch (error) {
        throw new AppError(
          "Failed to list customer orders",
          "ORDERS_LIST_ERROR",
          error
        );
      }
    },
  }),

  getPaymentDetails: tool({
    description: "Get payment details for a specific order",
    inputSchema: z.object({
      orderId: z
        .string()
        .min(1, "Invalid order ID format")
        .describe("The ID of the order"),
    }),
    execute: async ({ orderId }) => {
      try {
        logger.debug({ orderId }, "Fetching payment details");
        const { order } = await medusa.admin.orders.retrieve(orderId);

        if (!order) {
          throw new AppError("Order not found", "ORDER_NOT_FOUND");
        }

        logger.debug({ orderId }, "Successfully fetched payment details");
        return order.payments.map((payment) => ({
          id: payment.id,
          amount: payment.amount,
          currency: payment.currency_code,
          provider: payment.provider_id,
          status: payment.status,
          createdAt: payment.created_at,
        }));
      } catch (error) {
        if (error.status === 404) {
          throw new AppError("Order not found", "ORDER_NOT_FOUND", error);
        }
        throw new AppError(
          "Failed to fetch payment details",
          "PAYMENT_FETCH_ERROR",
          error
        );
      }
    },
  }),

  listCarts: tool({
    description: "List all carts in the system or for a specific customer",
    inputSchema: z.object({
      customerId: z
        .string()
        .min(1, "Invalid customer ID format")
        .optional()
        .describe("The ID of the customer (optional)"),
      limit: z
        .number()
        .min(1)
        .max(100)
        .default(20)
        .describe("Maximum number of carts to return"),
      offset: z
        .number()
        .min(0)
        .default(0)
        .describe("Number of carts to skip for pagination"),
    }),
    execute: async ({ customerId, limit, offset }) => {
      try {
        logger.debug({ customerId }, "Listing carts");
        const params = customerId ? { customer_id: customerId } : {};
        const { carts } = await medusa.admin.carts.list({
          ...params,
          limit,
          offset,
        });

        logger.debug({ count: carts.length }, "Successfully listed carts");
        return carts.map((cart) => ({
          id: cart.id,
          customerId: cart.customer_id,
          email: cart.email,
          items: cart.items.length,
          total: cart.total,
          createdAt: cart.created_at,
          updatedAt: cart.updated_at,
        }));
      } catch (error) {
        throw new AppError("Failed to list carts", "CARTS_LIST_ERROR", error);
      }
    },
  }),

  getCartDetails: tool({
    description: "Get details of a specific cart",
    inputSchema: z.object({
      cartId: z
        .string()
        .min(1, "Invalid cart ID format")
        .describe("The ID of the cart"),
    }),
    execute: async ({ cartId }) => {
      try {
        logger.debug({ cartId }, "Fetching cart details");
        const { cart } = await medusa.admin.carts.retrieve(cartId);

        if (!cart) {
          throw new AppError("Cart not found", "CART_NOT_FOUND");
        }

        logger.debug({ cartId }, "Successfully fetched cart details");
        return {
          id: cart.id,
          customerId: cart.customer_id,
          email: cart.email,
          items: cart.items,
          total: cart.total,
          shippingAddress: cart.shipping_address,
          billingAddress: cart.billing_address,
          discounts: cart.discounts,
          giftCards: cart.gift_cards,
          createdAt: cart.created_at,
          updatedAt: cart.updated_at,
        };
      } catch (error) {
        if (error.status === 404) {
          throw new AppError("Cart not found", "CART_NOT_FOUND", error);
        }
        throw new AppError(
          "Failed to fetch cart details",
          "CART_FETCH_ERROR",
          error
        );
      }
    },
  }),

  getOrderFulfillment: tool({
    description: "Get fulfillment status and details for an order",
    inputSchema: z.object({
      orderId: z
        .string()
        .min(1, "Invalid order ID format")
        .describe("The ID of the order"),
    }),
    execute: async ({ orderId }) => {
      try {
        logger.debug({ orderId }, "Fetching order fulfillment details");
        const { order } = await medusa.admin.orders.retrieve(orderId);

        if (!order) {
          throw new AppError("Order not found", "ORDER_NOT_FOUND");
        }

        logger.debug(
          { orderId },
          "Successfully fetched order fulfillment details"
        );
        return order.fulfillments.map((fulfillment) => ({
          id: fulfillment.id,
          status: fulfillment.status,
          items: fulfillment.items,
          trackingNumbers: fulfillment.tracking_numbers,
          shippedAt: fulfillment.shipped_at,
          deliveredAt: fulfillment.delivered_at,
          createdAt: fulfillment.created_at,
        }));
      } catch (error) {
        if (error.status === 404) {
          throw new AppError("Order not found", "ORDER_NOT_FOUND", error);
        }
        throw new AppError(
          "Failed to fetch fulfillment details",
          "FULFILLMENT_FETCH_ERROR",
          error
        );
      }
    },
  }),

  getOrderReturns: tool({
    description: "Get return requests and status for an order",
    inputSchema: z.object({
      orderId: z
        .string()
        .min(1, "Invalid order ID format")
        .describe("The ID of the order"),
    }),
    execute: async ({ orderId }) => {
      try {
        logger.debug({ orderId }, "Fetching order returns");
        const { order } = await medusa.admin.orders.retrieve(orderId);

        if (!order) {
          throw new AppError("Order not found", "ORDER_NOT_FOUND");
        }

        logger.debug({ orderId }, "Successfully fetched order returns");
        return order.returns.map((returnItem) => ({
          id: returnItem.id,
          status: returnItem.status,
          items: returnItem.items,
          reason: returnItem.reason,
          note: returnItem.note,
          requestedAt: returnItem.requested_at,
          receivedAt: returnItem.received_at,
          refundedAt: returnItem.refunded_at,
        }));
      } catch (error) {
        if (error.status === 404) {
          throw new AppError("Order not found", "ORDER_NOT_FOUND", error);
        }
        throw new AppError(
          "Failed to fetch order returns",
          "RETURNS_FETCH_ERROR",
          error
        );
      }
    },
  }),

  getOrderTimeline: tool({
    description:
      "Get the timeline of events for an order (created, paid, fulfilled, etc.)",
    inputSchema: z.object({
      orderId: z
        .string()
        .min(1, "Invalid order ID format")
        .describe("The ID of the order"),
    }),
    execute: async ({ orderId }) => {
      try {
        logger.debug({ orderId }, "Fetching order timeline");
        const { order } = await medusa.admin.orders.retrieve(orderId);

        if (!order) {
          throw new AppError("Order not found", "ORDER_NOT_FOUND");
        }

        logger.debug({ orderId }, "Successfully fetched order timeline");
        return {
          createdAt: order.created_at,
          paidAt: order.paid_at,
          fulfilledAt: order.fulfilled_at,
          canceledAt: order.canceled_at,
          returnedAt: order.returned_at,
          refundedAt: order.refunded_at,
          status: order.status,
        };
      } catch (error) {
        if (error.status === 404) {
          throw new AppError("Order not found", "ORDER_NOT_FOUND", error);
        }
        throw new AppError(
          "Failed to fetch order timeline",
          "TIMELINE_FETCH_ERROR",
          error
        );
      }
    },
  }),

  searchProducts: tool({
    description: "Search for products by name, description, or other criteria",
    inputSchema: z.object({
      query: z
        .string()
        .min(1, "Search query is required")
        .describe("Search query to find products"),
      limit: z
        .number()
        .min(1)
        .max(100)
        .default(10)
        .describe("Maximum number of products to return"),
    }),
    execute: async ({ query, limit }) => {
      try {
        logger.debug({ query }, "Searching products");
        const { products } = await medusa.admin.products.list({
          q: query,
          limit,
        });

        logger.debug(
          { query, count: products.length },
          "Product search completed"
        );
        return products.map((product) => ({
          id: product.id,
          title: product.title,
          description: product.description,
          thumbnail: product.thumbnail,
          variants: product.variants.length,
          status: product.status,
        }));
      } catch (error) {
        throw new AppError(
          "Failed to search products",
          "PRODUCT_SEARCH_ERROR",
          error
        );
      }
    },
  }),

  updateSupportTicket: tool({
    description: "Update an existing support ticket",
    inputSchema: z.object({
      ticketId: z
        .string()
        .min(1, "Invalid ticket ID format")
        .describe("The ID of the support ticket to update"),
      status: z
        .enum(["open", "in_progress", "resolved", "closed"])
        .optional()
        .describe("New status for the ticket"),
      priority: z
        .enum(["low", "medium", "high"])
        .optional()
        .describe("New priority for the ticket"),
      assigneeId: z
        .string()
        .min(1, "Invalid user ID format")
        .optional()
        .describe("ID of the agent to assign the ticket to"),
      note: z
        .string()
        .optional()
        .describe("Additional note to add to the ticket"),
    }),
    execute: async ({ ticketId, status, priority, assigneeId, note }) => {
      try {
        logger.debug({ ticketId }, "Updating support ticket");

        const updates = {
          updated_at: new Date().toISOString(),
        };

        if (status) updates.status = status;
        if (priority) updates.priority = priority;
        if (assigneeId) updates.assignee_id = assigneeId;

        const { data, error } = await supabase
          .from("support_tickets")
          .update(updates)
          .eq("id", ticketId)
          .select();

        if (error) {
          throw new AppError(
            "Database error updating ticket",
            "TICKET_UPDATE_ERROR",
            error
          );
        }

        // Add note if provided
        if (note) {
          await supabase.from("ticket_notes").insert([
            {
              ticket_id: ticketId,
              note,
              created_at: new Date().toISOString(),
            },
          ]);
        }

        logger.debug({ ticketId }, "Support ticket updated successfully");
        return data[0];
      } catch (error) {
        throw new AppError(
          "Failed to update support ticket",
          "TICKET_UPDATE_ERROR",
          error
        );
      }
    },
  }),

  createDiscount: tool({
    description: "Create a discount code for a customer",
    inputSchema: z.object({
      code: z
        .string()
        .min(3, "Discount code must be at least 3 characters")
        .describe("The discount code to create"),
      value: z
        .number()
        .min(1, "Discount value must be at least 1")
        .describe("The value of the discount"),
      type: z
        .enum(["fixed", "percentage"])
        .default("fixed")
        .describe("Type of discount (fixed amount or percentage)"),
      customerId: z
        .string()
        .min(1, "Invalid customer ID format")
        .optional()
        .describe("ID of the customer this discount is for (optional)"),
      expiresAt: z
        .string()
        .datetime()
        .optional()
        .describe("Expiration date for the discount (ISO format)"),
    }),
    execute: async ({ code, value, type, customerId, expiresAt }) => {
      try {
        logger.debug({ code }, "Creating discount");

        const discountData = {
          code,
          rule: {
            type,
            value,
            allocation: type === "fixed" ? "total" : "item",
          },
          is_dynamic: false,
          is_disabled: false,
        };

        if (customerId) {
          discountData.rule.conditions = [
            {
              operator: "in",
              customers: [customerId],
            },
          ];
        }

        if (expiresAt) {
          discountData.ends_at = expiresAt;
        }

        const { discount } = await medusa.admin.discounts.create(discountData);

        logger.debug(
          { code, discountId: discount.id },
          "Discount created successfully"
        );
        return {
          id: discount.id,
          code: discount.code,
          value: discount.rule.value,
          type: discount.rule.type,
          expiresAt: discount.ends_at,
          isDisabled: discount.is_disabled,
        };
      } catch (error) {
        throw new AppError(
          "Failed to create discount",
          "DISCOUNT_CREATION_ERROR",
          error
        );
      }
    },
  }),

  

  applyDiscountToCart: tool({
    description: "Apply a discount code to a customer's cart",
    inputSchema: z.object({
      cartId: z
        .string()
        .min(1, "Invalid cart ID format")
        .describe("The ID of the cart"),
      discountCode: z
        .string()
        .min(1, "Discount code is required")
        .describe("The discount code to apply"),
    }),
    execute: async ({ cartId, discountCode }) => {
      try {
        logger.debug({ cartId, discountCode }, "Applying discount to cart");

        const { cart } = await medusa.carts.update(cartId, {
          discounts: [{ code: discountCode }],
        });

        logger.debug(
          { cartId, discountCode },
          "Discount applied to cart successfully"
        );
        return {
          success: true,
          cartId: cart.id,
          discountCode,
          newTotal: cart.total,
          previousTotal: cart.subtotal,
        };
      } catch (error) {
        console.log(error);
        if (error.status === 404) {
          throw new AppError("Cart not found", "CART_NOT_FOUND", error);
        }
        throw new AppError(
          "Failed to apply discount to cart",
          "DISCOUNT_APPLICATION_ERROR",
          error
        );
      }
    },
  }),
};

// AI response generator with enhanced capabilities
class AIResponseGenerator {
  constructor() {
    this.providers = {
      deepseek: {
        generate: async (prompt, systemPrompt, tools) => {
          console.log("Deepseek model:", config.ai.deepseekModel);
          return await generateText({
            model: deepseek(config.ai.deepseekModel),
            system: systemPrompt,
            tools: tools,
            prompt: prompt,
            stopWhen: stepCountIs(20),
          });
        },
      },
      ollama: {
        generate: async (prompt, systemPrompt) => {
          const response = await ollama.generate({
            model: config.ai.model,
            prompt: prompt,
            stream: false,
            system: systemPrompt,
            temperature: 0.7,
          });
          return { text: response.response };
        },
      },
    };
  }

  async generateResponse(
    prompt,
    context = [],
    sessionId = null,
    mode = "support"
  ) {
    if (!prompt || typeof prompt !== "string") {
      throw new AppError("Invalid prompt provided", "INVALID_PROMPT");
    }

    // Check rate limiting
    if (!rateLimiter.check(sessionId || "anonymous")) {
      throw new AppError("Rate limit exceeded", "RATE_LIMIT_EXCEEDED");
    }

    try {
      const contextText = Array.isArray(context) ? context.join("\n") : context;
      const session = sessionId ? sessionManager.getSession(sessionId) : null;
      const systemPrompt = systemPrompts[mode];

      // Add customer context if available
      let customerContext = "";
      if (session?.customer) {
        customerContext = `Customer information: ${JSON.stringify(
          session.customer
        )}`;
      }

      console.log("Customer Context:", customerContext);

      const fullPrompt = `${customerContext}\n\nContext: ${contextText}\n\nUser: ${prompt}`;
      const provider =
        config.ai.provider === "deepseek" ? "deepseek" : "ollama";

      const { text, toolCalls, toolResults } = await this.providers[
        provider
      ].generate(
        fullPrompt,
        systemPrompt,
        config.ai.provider === "deepseek" ? tools : undefined
      );

      logger.debug(
        { sessionId, toolCalls, toolResults },
        "AI response generated"
      );

      return text;
    } catch (error) {
      // console.log("Error generating AI response:", error);
      logger.error({ error, sessionId }, "Error generating AI response");
      throw new AppError(
        "Failed to generate AI response",
        "AI_GENERATION_ERROR",
        error
      );
    }
  }
}

const aiGenerator = new AIResponseGenerator();

// Conversation analysis with improved error handling
async function analyzeConversation(sessionId, messages) {
  try {
    const conversation = messages
      .map((m) => `${m.sender === "user" ? "User" : "Agent"}: ${m.message}`)
      .join("\n");

    const analysisPrompt = `
      Analyze this customer support conversation. 
      Return ONLY valid JSON with these exact keys: 
      "issues" (array of strings), 
      "suggestions" (array of strings), 
      "tone" (string), 
      "urgency" (string: "low" | "medium" | "high"). 
      Focus on identifying key issues, suggesting improvements, and assessing the tone and urgency.
      Conversation:
      ${conversation}
    `;

    const analysis = await aiGenerator.generateResponse(
      analysisPrompt,
      [],
      sessionId
    );
    const parsed = safeJSONParse(analysis);

    if (!parsed) {
      logger.warn(
        { sessionId, analysis },
        "Failed to parse conversation analysis"
      );
      return {
        issues: ["Unable to analyze conversation"],
        suggestions: ["Please review manually"],
        tone: "neutral",
        urgency: "medium",
      };
    }

    return parsed;
  } catch (error) {
    logger.error({ error, sessionId }, "Error analyzing conversation");
    return null;
  }
}

// Utility functions
function safeJSONParse(str) {
  try {
    // Handle cases where AI might wrap JSON in markdown code blocks
    const jsonStr = str.replace(/```json\n?|\n?```/g, "");
    return JSON.parse(jsonStr);
  } catch (e) {
    return null;
  }
}

// Channel setup and event handlers
function setupChannels() {
  const aiChannel = supabase.channel("ai_requests", {
    config: {
      presence: {
        key: "ai_service",
      },
    },
  });

  // AI request handler
  aiChannel
    .on("presence", { event: "sync" }, () => {
      logger.debug("Online AI service:", Array.from(aiChannel.presenceState()));
    })
    .on("broadcast", { event: "ai_request" }, async (payload) => {
      const { sessionId, message, context } = payload.payload;
      logger.info({ sessionId }, "AI request received");

      try {
        // Initialize session if not exists
        if (!sessionManager.getSession(sessionId)) {
          await initializeSession(sessionId);
        }

        sessionManager.updateSessionActivity(sessionId);

        // Generate AI response
        const session = sessionManager.getSession(sessionId);
        const aiResponse = await aiGenerator.generateResponse(
          message,
          context,
          sessionId,
          session.mode
        );

        // Send response back
        await sendSessionMessage(sessionId, "ai_response", {
          message: aiResponse,
        });

        logger.debug({ sessionId }, "AI response sent");
      } catch (error) {
        logger.error({ error, sessionId }, "Error processing AI request");

        // Send error response
        await sendSessionMessage(sessionId, "ai_error", {
          error: error.message,
          code: error.code,
        });
      }
    })
    .subscribe((status) => {
      if (status === "SUBSCRIBED") {
        logger.info("AI service subscribed to ai_requests channel");
        aiChannel.track({
          online_at: new Date().toISOString(),
          service: "ai_assistant",
        });
      }
    });

  // Setup session cleanup interval
  setInterval(() => {
    sessionManager.cleanupInactiveSessions();
  }, config.session.cleanupInterval);

  return aiChannel;
}

// Helper functions
async function initializeSession(sessionId) {
  try {
    let customerInfo = null;

    // Try to get session data from database
    const { data: sessionData, error: sessionError } = await supabase
      .from("support_session")
      .select("*")
      .eq("id", sessionId)
      .single();

    if (!sessionError && sessionData && sessionData.user_id) {
      try {
        const { customer } = await medusa.admin.customers.retrieve(
          sessionData.user_id
        );
        customerInfo = {
          id: customer.id,
          email: customer.email,
          name: `${customer.first_name} ${customer.last_name}`,
        };
      } catch (error) {
        logger.warn(
          { error, sessionId, userId: sessionData.user_id },
          "Failed to fetch customer info"
        );
      }
    }

    sessionManager.setSession(sessionId, {
      customer: customerInfo,
      mode: "support",
      createdAt: new Date().toISOString(),
    });

    logger.debug({ sessionId }, "Session initialized");
  } catch (error) {
    logger.error({ error, sessionId }, "Error initializing session");
    // Create a basic session even if initialization partially fails
    sessionManager.setSession(sessionId, {
      customer: null,
      mode: "support",
      createdAt: new Date().toISOString(),
    });
  }
}

async function sendSessionMessage(sessionId, event, payload) {
  try {
    let channel = sessionManager.getChannel(sessionId);

    if (!channel) {
      channel = supabase.channel(`session:${sessionId}`);
      sessionManager.setChannel(sessionId, channel);
      await channel.subscribe();
    }

    await channel.send({
      type: "broadcast",
      event: event,
      payload: {
        type: event,
        sessionId: sessionId,
        timestamp: new Date().toISOString(),
        ...payload,
      },
    });
  } catch (error) {
    logger.error({ error, sessionId, event }, "Error sending session message");
  }
}

// Health check endpoint
function setupHealthCheck() {
  const healthChannel = supabase.channel("health", {
    config: {
      presence: {
        key: "health_check",
      },
    },
  });

  healthChannel
    .on("broadcast", { event: "health_check" }, async (payload) => {
      try {
        // Check database connection
        await supabase.from("support_session").select("count").limit(1);

        // Check Medusa connection
        await medusa.admin.products.list({ limit: 1 });

        healthChannel.send({
          type: "broadcast",
          event: "health_response",
          payload: {
            status: "healthy",
            timestamp: new Date().toISOString(),
            services: {
              supabase: "connected",
              medusa: "connected",
              ai: config.ai.provider,
            },
          },
        });
      } catch (error) {
        healthChannel.send({
          type: "broadcast",
          event: "health_response",
          payload: {
            status: "unhealthy",
            timestamp: new Date().toISOString(),
            error: error.message,
          },
        });
      }
    })
    .subscribe();

  return healthChannel;
}

// Update the main initialization to include haggle manager
async function initialize() {
  try {
    logger.info("Starting AI service...");

    // Test connections
    await supabase.from("support_sessions").select("count").limit(1);
    await medusa.admin.products.list({ limit: 1 });

    // Setup channels
    const aiChannel = setupChannels();
    const healthChannel = setupHealthCheck();

    // Initialize haggle manager
    haggleManager.initializeHagglePresence();

    logger.info("AI service started successfully");


    // Setup session cleanup interval for haggle sessions too
    setInterval(() => {
      sessionManager.cleanupInactiveSessions();
      haggleManager.cleanupInactiveSessions();
    }, config.session.cleanupInterval);

    // Graceful shutdown handler
    const shutdown = async (signal) => {
      logger.info(`Received ${signal}, shutting down gracefully...`);

      // Cleanup channels
      for (const [sessionId, channel] of sessionManager.channels.entries()) {
        await supabase.removeChannel(channel);
      }

      // Cleanup haggle sessions
      for (const sessionId of haggleManager.getActiveSessions()) {
        await haggleManager.removeHaggleSession(sessionId);
      }

      await supabase.removeChannel(aiChannel);
      await supabase.removeChannel(healthChannel);

      if (haggleManager.presenceChannel) {
        await supabase.removeChannel(haggleManager.presenceChannel);
      }

      logger.info("Cleanup completed, exiting");
      process.exit(0);
    };

    // Handle graceful shutdown
    process.on("SIGINT", () => shutdown("SIGINT"));
    process.on("SIGTERM", () => shutdown("SIGTERM"));
  } catch (error) {
    logger.error({ error }, "Failed to initialize AI service");
    process.exit(1);
  }
}

// Start the service
initialize().catch((error) => {
  logger.error({ error }, "Unhandled error during initialization");
  process.exit(1);
});

export {
  aiGenerator,
  sessionManager,
  tools,
  RateLimiter,
  AppError,
  handleError,
};
