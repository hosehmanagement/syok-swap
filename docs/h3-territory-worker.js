// DOCS ONLY - NOT DEPLOYED. Example Cloudflare Worker check for a possible MiniMax H3 engine route.
// A safeguard written with section V.5 of the H3 Community License in mind; it does NOT make anything legally compliant.
// The country is Cloudflare's guess from the visitor's IP address: a VPN or proxy gets around it. Needs a lawyer's review.
// The list must match BLOCKED_COUNTRY_CODES in h3_territory.py (tests/test_h3_territory.py checks this file).

const BLOCKED = new Set([
  "US", "GB", "KR",
  // the 27 EU member states
  "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE",
  "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE",
]);
// Not a country: XX = Cloudflare could not tell, T1 = Tor exit node. Treated as blocked (fail closed).
const NOT_A_COUNTRY = new Set(["XX", "T1", "ZZ", "EU", "AP"]);

const BLOCKED_MESSAGE =
  "MiniMax H3 is not available in your region. Its licence does not allow use in the US, the EU, the UK or " +
  "South Korea, and we could not confirm you are outside them. You can still use the standard engine.";

export function h3Blocked(country) {
  if (typeof country !== "string") return true;            // missing or unknown: blocked
  const c = country.trim().toUpperCase();
  if (!/^[A-Z]{2}$/.test(c)) return true;
  if (NOT_A_COUNTRY.has(c)) return true;
  return BLOCKED.has(c);
}

export default {
  async fetch(request, env, ctx) {
    const country = request.cf && request.cf.country;
    if (h3Blocked(country)) {
      return new Response(JSON.stringify({ error: BLOCKED_MESSAGE }), {
        status: 403,
        headers: { "content-type": "application/json", "cache-control": "no-store" },
      });
    }
    // Allowed: pass the request on to the H3 engine (not written here).
    return fetch(request);
  },
};
