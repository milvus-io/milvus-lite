
// Generated from Plan.g4 by ANTLR 4.13.1


#include "PlanVisitor.h"

#include "PlanParser.h"


using namespace antlrcpp;

using namespace antlr4;

namespace {

struct PlanParserStaticData final {
  PlanParserStaticData(std::vector<std::string> ruleNames,
                        std::vector<std::string> literalNames,
                        std::vector<std::string> symbolicNames)
      : ruleNames(std::move(ruleNames)), literalNames(std::move(literalNames)),
        symbolicNames(std::move(symbolicNames)),
        vocabulary(this->literalNames, this->symbolicNames) {}

  PlanParserStaticData(const PlanParserStaticData&) = delete;
  PlanParserStaticData(PlanParserStaticData&&) = delete;
  PlanParserStaticData& operator=(const PlanParserStaticData&) = delete;
  PlanParserStaticData& operator=(PlanParserStaticData&&) = delete;

  std::vector<antlr4::dfa::DFA> decisionToDFA;
  antlr4::atn::PredictionContextCache sharedContextCache;
  const std::vector<std::string> ruleNames;
  const std::vector<std::string> literalNames;
  const std::vector<std::string> symbolicNames;
  const antlr4::dfa::Vocabulary vocabulary;
  antlr4::atn::SerializedATNView serializedATN;
  std::unique_ptr<antlr4::atn::ATN> atn;
};

::antlr4::internal::OnceFlag planParserOnceFlag;
#if ANTLR4_USE_THREAD_LOCAL_CACHE
static thread_local
#endif
PlanParserStaticData *planParserStaticData = nullptr;

void planParserInitialize() {
#if ANTLR4_USE_THREAD_LOCAL_CACHE
  if (planParserStaticData != nullptr) {
    return;
  }
#else
  assert(planParserStaticData == nullptr);
#endif
  auto staticData = std::make_unique<PlanParserStaticData>(
    std::vector<std::string>{
      "expr"
    },
    std::vector<std::string>{
      "", "'('", "')'", "'['", "','", "']'", "'<'", "'<='", "'>'", "'>='", 
      "'=='", "'!='", "", "", "'+'", "'-'", "'*'", "'/'", "'%'", "'**'", 
      "'<<'", "'>>'", "'&'", "'|'", "'^'", "", "", "'~'", "", "'in'", "'not in'"
    },
    std::vector<std::string>{
      "", "", "", "", "", "", "LT", "LE", "GT", "GE", "EQ", "NE", "LIKE", 
      "EXISTS", "ADD", "SUB", "MUL", "DIV", "MOD", "POW", "SHL", "SHR", 
      "BAND", "BOR", "BXOR", "AND", "OR", "BNOT", "NOT", "IN", "NIN", "EmptyTerm", 
      "JSONContains", "JSONContainsAll", "JSONContainsAny", "ArrayContains", 
      "ArrayContainsAll", "ArrayContainsAny", "ArrayLength", "BooleanConstant", 
      "IntegerConstant", "FloatingConstant", "Identifier", "StringLiteral", 
      "JSONIdentifier", "Whitespace", "Newline"
    }
  );
  static const int32_t serializedATNSegment[] = {
  	4,1,46,129,2,0,7,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,
  	0,1,0,1,0,5,0,18,8,0,10,0,12,0,21,9,0,1,0,3,0,24,8,0,1,0,1,0,1,0,1,0,
  	1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,
  	0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,3,0,57,8,0,1,0,1,0,1,0,1,0,1,0,
  	1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,
  	0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,
  	1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,5,0,111,8,0,10,0,12,0,
  	114,9,0,1,0,3,0,117,8,0,1,0,1,0,1,0,1,0,1,0,5,0,124,8,0,10,0,12,0,127,
  	9,0,1,0,0,1,0,1,0,0,13,2,0,14,15,27,28,2,0,32,32,35,35,2,0,33,33,36,36,
  	2,0,34,34,37,37,2,0,42,42,44,44,1,0,16,18,1,0,14,15,1,0,20,21,1,0,6,7,
  	1,0,8,9,1,0,6,9,1,0,10,11,1,0,29,30,160,0,56,1,0,0,0,2,3,6,0,-1,0,3,57,
  	5,40,0,0,4,57,5,41,0,0,5,57,5,39,0,0,6,57,5,43,0,0,7,57,5,42,0,0,8,57,
  	5,44,0,0,9,10,5,1,0,0,10,11,3,0,0,0,11,12,5,2,0,0,12,57,1,0,0,0,13,14,
  	5,3,0,0,14,19,3,0,0,0,15,16,5,4,0,0,16,18,3,0,0,0,17,15,1,0,0,0,18,21,
  	1,0,0,0,19,17,1,0,0,0,19,20,1,0,0,0,20,23,1,0,0,0,21,19,1,0,0,0,22,24,
  	5,4,0,0,23,22,1,0,0,0,23,24,1,0,0,0,24,25,1,0,0,0,25,26,5,5,0,0,26,57,
  	1,0,0,0,27,28,7,0,0,0,28,57,3,0,0,20,29,30,7,1,0,0,30,31,5,1,0,0,31,32,
  	3,0,0,0,32,33,5,4,0,0,33,34,3,0,0,0,34,35,5,2,0,0,35,57,1,0,0,0,36,37,
  	7,2,0,0,37,38,5,1,0,0,38,39,3,0,0,0,39,40,5,4,0,0,40,41,3,0,0,0,41,42,
  	5,2,0,0,42,57,1,0,0,0,43,44,7,3,0,0,44,45,5,1,0,0,45,46,3,0,0,0,46,47,
  	5,4,0,0,47,48,3,0,0,0,48,49,5,2,0,0,49,57,1,0,0,0,50,51,5,38,0,0,51,52,
  	5,1,0,0,52,53,7,4,0,0,53,57,5,2,0,0,54,55,5,13,0,0,55,57,3,0,0,1,56,2,
  	1,0,0,0,56,4,1,0,0,0,56,5,1,0,0,0,56,6,1,0,0,0,56,7,1,0,0,0,56,8,1,0,
  	0,0,56,9,1,0,0,0,56,13,1,0,0,0,56,27,1,0,0,0,56,29,1,0,0,0,56,36,1,0,
  	0,0,56,43,1,0,0,0,56,50,1,0,0,0,56,54,1,0,0,0,57,125,1,0,0,0,58,59,10,
  	21,0,0,59,60,5,19,0,0,60,124,3,0,0,22,61,62,10,19,0,0,62,63,7,5,0,0,63,
  	124,3,0,0,20,64,65,10,18,0,0,65,66,7,6,0,0,66,124,3,0,0,19,67,68,10,17,
  	0,0,68,69,7,7,0,0,69,124,3,0,0,18,70,71,10,10,0,0,71,72,7,8,0,0,72,73,
  	7,4,0,0,73,74,7,8,0,0,74,124,3,0,0,11,75,76,10,9,0,0,76,77,7,9,0,0,77,
  	78,7,4,0,0,78,79,7,9,0,0,79,124,3,0,0,10,80,81,10,8,0,0,81,82,7,10,0,
  	0,82,124,3,0,0,9,83,84,10,7,0,0,84,85,7,11,0,0,85,124,3,0,0,8,86,87,10,
  	6,0,0,87,88,5,22,0,0,88,124,3,0,0,7,89,90,10,5,0,0,90,91,5,24,0,0,91,
  	124,3,0,0,6,92,93,10,4,0,0,93,94,5,23,0,0,94,124,3,0,0,5,95,96,10,3,0,
  	0,96,97,5,25,0,0,97,124,3,0,0,4,98,99,10,2,0,0,99,100,5,26,0,0,100,124,
  	3,0,0,3,101,102,10,22,0,0,102,103,5,12,0,0,103,124,5,43,0,0,104,105,10,
  	16,0,0,105,106,7,12,0,0,106,107,5,3,0,0,107,112,3,0,0,0,108,109,5,4,0,
  	0,109,111,3,0,0,0,110,108,1,0,0,0,111,114,1,0,0,0,112,110,1,0,0,0,112,
  	113,1,0,0,0,113,116,1,0,0,0,114,112,1,0,0,0,115,117,5,4,0,0,116,115,1,
  	0,0,0,116,117,1,0,0,0,117,118,1,0,0,0,118,119,5,5,0,0,119,124,1,0,0,0,
  	120,121,10,15,0,0,121,122,7,12,0,0,122,124,5,31,0,0,123,58,1,0,0,0,123,
  	61,1,0,0,0,123,64,1,0,0,0,123,67,1,0,0,0,123,70,1,0,0,0,123,75,1,0,0,
  	0,123,80,1,0,0,0,123,83,1,0,0,0,123,86,1,0,0,0,123,89,1,0,0,0,123,92,
  	1,0,0,0,123,95,1,0,0,0,123,98,1,0,0,0,123,101,1,0,0,0,123,104,1,0,0,0,
  	123,120,1,0,0,0,124,127,1,0,0,0,125,123,1,0,0,0,125,126,1,0,0,0,126,1,
  	1,0,0,0,127,125,1,0,0,0,7,19,23,56,112,116,123,125
  };
  staticData->serializedATN = antlr4::atn::SerializedATNView(serializedATNSegment, sizeof(serializedATNSegment) / sizeof(serializedATNSegment[0]));

  antlr4::atn::ATNDeserializer deserializer;
  staticData->atn = deserializer.deserialize(staticData->serializedATN);

  const size_t count = staticData->atn->getNumberOfDecisions();
  staticData->decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    staticData->decisionToDFA.emplace_back(staticData->atn->getDecisionState(i), i);
  }
  planParserStaticData = staticData.release();
}

}

PlanParser::PlanParser(TokenStream *input) : PlanParser(input, antlr4::atn::ParserATNSimulatorOptions()) {}

PlanParser::PlanParser(TokenStream *input, const antlr4::atn::ParserATNSimulatorOptions &options) : Parser(input) {
  PlanParser::initialize();
  _interpreter = new atn::ParserATNSimulator(this, *planParserStaticData->atn, planParserStaticData->decisionToDFA, planParserStaticData->sharedContextCache, options);
}

PlanParser::~PlanParser() {
  delete _interpreter;
}

const atn::ATN& PlanParser::getATN() const {
  return *planParserStaticData->atn;
}

std::string PlanParser::getGrammarFileName() const {
  return "Plan.g4";
}

const std::vector<std::string>& PlanParser::getRuleNames() const {
  return planParserStaticData->ruleNames;
}

const dfa::Vocabulary& PlanParser::getVocabulary() const {
  return planParserStaticData->vocabulary;
}

antlr4::atn::SerializedATNView PlanParser::getSerializedATN() const {
  return planParserStaticData->serializedATN;
}


//----------------- ExprContext ------------------------------------------------------------------

PlanParser::ExprContext::ExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t PlanParser::ExprContext::getRuleIndex() const {
  return PlanParser::RuleExpr;
}

void PlanParser::ExprContext::copyFrom(ExprContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- JSONIdentifierContext ------------------------------------------------------------------

tree::TerminalNode* PlanParser::JSONIdentifierContext::JSONIdentifier() {
  return getToken(PlanParser::JSONIdentifier, 0);
}

PlanParser::JSONIdentifierContext::JSONIdentifierContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::JSONIdentifierContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitJSONIdentifier(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ParensContext ------------------------------------------------------------------

PlanParser::ExprContext* PlanParser::ParensContext::expr() {
  return getRuleContext<PlanParser::ExprContext>(0);
}

PlanParser::ParensContext::ParensContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::ParensContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitParens(this);
  else
    return visitor->visitChildren(this);
}
//----------------- StringContext ------------------------------------------------------------------

tree::TerminalNode* PlanParser::StringContext::StringLiteral() {
  return getToken(PlanParser::StringLiteral, 0);
}

PlanParser::StringContext::StringContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::StringContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitString(this);
  else
    return visitor->visitChildren(this);
}
//----------------- FloatingContext ------------------------------------------------------------------

tree::TerminalNode* PlanParser::FloatingContext::FloatingConstant() {
  return getToken(PlanParser::FloatingConstant, 0);
}

PlanParser::FloatingContext::FloatingContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::FloatingContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitFloating(this);
  else
    return visitor->visitChildren(this);
}
//----------------- JSONContainsAllContext ------------------------------------------------------------------

std::vector<PlanParser::ExprContext *> PlanParser::JSONContainsAllContext::expr() {
  return getRuleContexts<PlanParser::ExprContext>();
}

PlanParser::ExprContext* PlanParser::JSONContainsAllContext::expr(size_t i) {
  return getRuleContext<PlanParser::ExprContext>(i);
}

tree::TerminalNode* PlanParser::JSONContainsAllContext::JSONContainsAll() {
  return getToken(PlanParser::JSONContainsAll, 0);
}

tree::TerminalNode* PlanParser::JSONContainsAllContext::ArrayContainsAll() {
  return getToken(PlanParser::ArrayContainsAll, 0);
}

PlanParser::JSONContainsAllContext::JSONContainsAllContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::JSONContainsAllContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitJSONContainsAll(this);
  else
    return visitor->visitChildren(this);
}
//----------------- LogicalOrContext ------------------------------------------------------------------

std::vector<PlanParser::ExprContext *> PlanParser::LogicalOrContext::expr() {
  return getRuleContexts<PlanParser::ExprContext>();
}

PlanParser::ExprContext* PlanParser::LogicalOrContext::expr(size_t i) {
  return getRuleContext<PlanParser::ExprContext>(i);
}

tree::TerminalNode* PlanParser::LogicalOrContext::OR() {
  return getToken(PlanParser::OR, 0);
}

PlanParser::LogicalOrContext::LogicalOrContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::LogicalOrContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitLogicalOr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- MulDivModContext ------------------------------------------------------------------

std::vector<PlanParser::ExprContext *> PlanParser::MulDivModContext::expr() {
  return getRuleContexts<PlanParser::ExprContext>();
}

PlanParser::ExprContext* PlanParser::MulDivModContext::expr(size_t i) {
  return getRuleContext<PlanParser::ExprContext>(i);
}

tree::TerminalNode* PlanParser::MulDivModContext::MUL() {
  return getToken(PlanParser::MUL, 0);
}

tree::TerminalNode* PlanParser::MulDivModContext::DIV() {
  return getToken(PlanParser::DIV, 0);
}

tree::TerminalNode* PlanParser::MulDivModContext::MOD() {
  return getToken(PlanParser::MOD, 0);
}

PlanParser::MulDivModContext::MulDivModContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::MulDivModContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitMulDivMod(this);
  else
    return visitor->visitChildren(this);
}
//----------------- IdentifierContext ------------------------------------------------------------------

tree::TerminalNode* PlanParser::IdentifierContext::Identifier() {
  return getToken(PlanParser::Identifier, 0);
}

PlanParser::IdentifierContext::IdentifierContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::IdentifierContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitIdentifier(this);
  else
    return visitor->visitChildren(this);
}
//----------------- LikeContext ------------------------------------------------------------------

PlanParser::ExprContext* PlanParser::LikeContext::expr() {
  return getRuleContext<PlanParser::ExprContext>(0);
}

tree::TerminalNode* PlanParser::LikeContext::LIKE() {
  return getToken(PlanParser::LIKE, 0);
}

tree::TerminalNode* PlanParser::LikeContext::StringLiteral() {
  return getToken(PlanParser::StringLiteral, 0);
}

PlanParser::LikeContext::LikeContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::LikeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitLike(this);
  else
    return visitor->visitChildren(this);
}
//----------------- LogicalAndContext ------------------------------------------------------------------

std::vector<PlanParser::ExprContext *> PlanParser::LogicalAndContext::expr() {
  return getRuleContexts<PlanParser::ExprContext>();
}

PlanParser::ExprContext* PlanParser::LogicalAndContext::expr(size_t i) {
  return getRuleContext<PlanParser::ExprContext>(i);
}

tree::TerminalNode* PlanParser::LogicalAndContext::AND() {
  return getToken(PlanParser::AND, 0);
}

PlanParser::LogicalAndContext::LogicalAndContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::LogicalAndContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitLogicalAnd(this);
  else
    return visitor->visitChildren(this);
}
//----------------- EqualityContext ------------------------------------------------------------------

std::vector<PlanParser::ExprContext *> PlanParser::EqualityContext::expr() {
  return getRuleContexts<PlanParser::ExprContext>();
}

PlanParser::ExprContext* PlanParser::EqualityContext::expr(size_t i) {
  return getRuleContext<PlanParser::ExprContext>(i);
}

tree::TerminalNode* PlanParser::EqualityContext::EQ() {
  return getToken(PlanParser::EQ, 0);
}

tree::TerminalNode* PlanParser::EqualityContext::NE() {
  return getToken(PlanParser::NE, 0);
}

PlanParser::EqualityContext::EqualityContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::EqualityContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitEquality(this);
  else
    return visitor->visitChildren(this);
}
//----------------- BooleanContext ------------------------------------------------------------------

tree::TerminalNode* PlanParser::BooleanContext::BooleanConstant() {
  return getToken(PlanParser::BooleanConstant, 0);
}

PlanParser::BooleanContext::BooleanContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::BooleanContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitBoolean(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ShiftContext ------------------------------------------------------------------

std::vector<PlanParser::ExprContext *> PlanParser::ShiftContext::expr() {
  return getRuleContexts<PlanParser::ExprContext>();
}

PlanParser::ExprContext* PlanParser::ShiftContext::expr(size_t i) {
  return getRuleContext<PlanParser::ExprContext>(i);
}

tree::TerminalNode* PlanParser::ShiftContext::SHL() {
  return getToken(PlanParser::SHL, 0);
}

tree::TerminalNode* PlanParser::ShiftContext::SHR() {
  return getToken(PlanParser::SHR, 0);
}

PlanParser::ShiftContext::ShiftContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::ShiftContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitShift(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ReverseRangeContext ------------------------------------------------------------------

std::vector<PlanParser::ExprContext *> PlanParser::ReverseRangeContext::expr() {
  return getRuleContexts<PlanParser::ExprContext>();
}

PlanParser::ExprContext* PlanParser::ReverseRangeContext::expr(size_t i) {
  return getRuleContext<PlanParser::ExprContext>(i);
}

tree::TerminalNode* PlanParser::ReverseRangeContext::Identifier() {
  return getToken(PlanParser::Identifier, 0);
}

tree::TerminalNode* PlanParser::ReverseRangeContext::JSONIdentifier() {
  return getToken(PlanParser::JSONIdentifier, 0);
}

std::vector<tree::TerminalNode *> PlanParser::ReverseRangeContext::GT() {
  return getTokens(PlanParser::GT);
}

tree::TerminalNode* PlanParser::ReverseRangeContext::GT(size_t i) {
  return getToken(PlanParser::GT, i);
}

std::vector<tree::TerminalNode *> PlanParser::ReverseRangeContext::GE() {
  return getTokens(PlanParser::GE);
}

tree::TerminalNode* PlanParser::ReverseRangeContext::GE(size_t i) {
  return getToken(PlanParser::GE, i);
}

PlanParser::ReverseRangeContext::ReverseRangeContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::ReverseRangeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitReverseRange(this);
  else
    return visitor->visitChildren(this);
}
//----------------- BitOrContext ------------------------------------------------------------------

std::vector<PlanParser::ExprContext *> PlanParser::BitOrContext::expr() {
  return getRuleContexts<PlanParser::ExprContext>();
}

PlanParser::ExprContext* PlanParser::BitOrContext::expr(size_t i) {
  return getRuleContext<PlanParser::ExprContext>(i);
}

tree::TerminalNode* PlanParser::BitOrContext::BOR() {
  return getToken(PlanParser::BOR, 0);
}

PlanParser::BitOrContext::BitOrContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::BitOrContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitBitOr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- AddSubContext ------------------------------------------------------------------

std::vector<PlanParser::ExprContext *> PlanParser::AddSubContext::expr() {
  return getRuleContexts<PlanParser::ExprContext>();
}

PlanParser::ExprContext* PlanParser::AddSubContext::expr(size_t i) {
  return getRuleContext<PlanParser::ExprContext>(i);
}

tree::TerminalNode* PlanParser::AddSubContext::ADD() {
  return getToken(PlanParser::ADD, 0);
}

tree::TerminalNode* PlanParser::AddSubContext::SUB() {
  return getToken(PlanParser::SUB, 0);
}

PlanParser::AddSubContext::AddSubContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::AddSubContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitAddSub(this);
  else
    return visitor->visitChildren(this);
}
//----------------- RelationalContext ------------------------------------------------------------------

std::vector<PlanParser::ExprContext *> PlanParser::RelationalContext::expr() {
  return getRuleContexts<PlanParser::ExprContext>();
}

PlanParser::ExprContext* PlanParser::RelationalContext::expr(size_t i) {
  return getRuleContext<PlanParser::ExprContext>(i);
}

tree::TerminalNode* PlanParser::RelationalContext::LT() {
  return getToken(PlanParser::LT, 0);
}

tree::TerminalNode* PlanParser::RelationalContext::LE() {
  return getToken(PlanParser::LE, 0);
}

tree::TerminalNode* PlanParser::RelationalContext::GT() {
  return getToken(PlanParser::GT, 0);
}

tree::TerminalNode* PlanParser::RelationalContext::GE() {
  return getToken(PlanParser::GE, 0);
}

PlanParser::RelationalContext::RelationalContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::RelationalContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitRelational(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ArrayLengthContext ------------------------------------------------------------------

tree::TerminalNode* PlanParser::ArrayLengthContext::ArrayLength() {
  return getToken(PlanParser::ArrayLength, 0);
}

tree::TerminalNode* PlanParser::ArrayLengthContext::Identifier() {
  return getToken(PlanParser::Identifier, 0);
}

tree::TerminalNode* PlanParser::ArrayLengthContext::JSONIdentifier() {
  return getToken(PlanParser::JSONIdentifier, 0);
}

PlanParser::ArrayLengthContext::ArrayLengthContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::ArrayLengthContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitArrayLength(this);
  else
    return visitor->visitChildren(this);
}
//----------------- TermContext ------------------------------------------------------------------

std::vector<PlanParser::ExprContext *> PlanParser::TermContext::expr() {
  return getRuleContexts<PlanParser::ExprContext>();
}

PlanParser::ExprContext* PlanParser::TermContext::expr(size_t i) {
  return getRuleContext<PlanParser::ExprContext>(i);
}

tree::TerminalNode* PlanParser::TermContext::IN() {
  return getToken(PlanParser::IN, 0);
}

tree::TerminalNode* PlanParser::TermContext::NIN() {
  return getToken(PlanParser::NIN, 0);
}

PlanParser::TermContext::TermContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::TermContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitTerm(this);
  else
    return visitor->visitChildren(this);
}
//----------------- JSONContainsContext ------------------------------------------------------------------

std::vector<PlanParser::ExprContext *> PlanParser::JSONContainsContext::expr() {
  return getRuleContexts<PlanParser::ExprContext>();
}

PlanParser::ExprContext* PlanParser::JSONContainsContext::expr(size_t i) {
  return getRuleContext<PlanParser::ExprContext>(i);
}

tree::TerminalNode* PlanParser::JSONContainsContext::JSONContains() {
  return getToken(PlanParser::JSONContains, 0);
}

tree::TerminalNode* PlanParser::JSONContainsContext::ArrayContains() {
  return getToken(PlanParser::ArrayContains, 0);
}

PlanParser::JSONContainsContext::JSONContainsContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::JSONContainsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitJSONContains(this);
  else
    return visitor->visitChildren(this);
}
//----------------- RangeContext ------------------------------------------------------------------

std::vector<PlanParser::ExprContext *> PlanParser::RangeContext::expr() {
  return getRuleContexts<PlanParser::ExprContext>();
}

PlanParser::ExprContext* PlanParser::RangeContext::expr(size_t i) {
  return getRuleContext<PlanParser::ExprContext>(i);
}

tree::TerminalNode* PlanParser::RangeContext::Identifier() {
  return getToken(PlanParser::Identifier, 0);
}

tree::TerminalNode* PlanParser::RangeContext::JSONIdentifier() {
  return getToken(PlanParser::JSONIdentifier, 0);
}

std::vector<tree::TerminalNode *> PlanParser::RangeContext::LT() {
  return getTokens(PlanParser::LT);
}

tree::TerminalNode* PlanParser::RangeContext::LT(size_t i) {
  return getToken(PlanParser::LT, i);
}

std::vector<tree::TerminalNode *> PlanParser::RangeContext::LE() {
  return getTokens(PlanParser::LE);
}

tree::TerminalNode* PlanParser::RangeContext::LE(size_t i) {
  return getToken(PlanParser::LE, i);
}

PlanParser::RangeContext::RangeContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::RangeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitRange(this);
  else
    return visitor->visitChildren(this);
}
//----------------- UnaryContext ------------------------------------------------------------------

PlanParser::ExprContext* PlanParser::UnaryContext::expr() {
  return getRuleContext<PlanParser::ExprContext>(0);
}

tree::TerminalNode* PlanParser::UnaryContext::ADD() {
  return getToken(PlanParser::ADD, 0);
}

tree::TerminalNode* PlanParser::UnaryContext::SUB() {
  return getToken(PlanParser::SUB, 0);
}

tree::TerminalNode* PlanParser::UnaryContext::BNOT() {
  return getToken(PlanParser::BNOT, 0);
}

tree::TerminalNode* PlanParser::UnaryContext::NOT() {
  return getToken(PlanParser::NOT, 0);
}

PlanParser::UnaryContext::UnaryContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::UnaryContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitUnary(this);
  else
    return visitor->visitChildren(this);
}
//----------------- IntegerContext ------------------------------------------------------------------

tree::TerminalNode* PlanParser::IntegerContext::IntegerConstant() {
  return getToken(PlanParser::IntegerConstant, 0);
}

PlanParser::IntegerContext::IntegerContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::IntegerContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitInteger(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ArrayContext ------------------------------------------------------------------

std::vector<PlanParser::ExprContext *> PlanParser::ArrayContext::expr() {
  return getRuleContexts<PlanParser::ExprContext>();
}

PlanParser::ExprContext* PlanParser::ArrayContext::expr(size_t i) {
  return getRuleContext<PlanParser::ExprContext>(i);
}

PlanParser::ArrayContext::ArrayContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::ArrayContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitArray(this);
  else
    return visitor->visitChildren(this);
}
//----------------- JSONContainsAnyContext ------------------------------------------------------------------

std::vector<PlanParser::ExprContext *> PlanParser::JSONContainsAnyContext::expr() {
  return getRuleContexts<PlanParser::ExprContext>();
}

PlanParser::ExprContext* PlanParser::JSONContainsAnyContext::expr(size_t i) {
  return getRuleContext<PlanParser::ExprContext>(i);
}

tree::TerminalNode* PlanParser::JSONContainsAnyContext::JSONContainsAny() {
  return getToken(PlanParser::JSONContainsAny, 0);
}

tree::TerminalNode* PlanParser::JSONContainsAnyContext::ArrayContainsAny() {
  return getToken(PlanParser::ArrayContainsAny, 0);
}

PlanParser::JSONContainsAnyContext::JSONContainsAnyContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::JSONContainsAnyContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitJSONContainsAny(this);
  else
    return visitor->visitChildren(this);
}
//----------------- BitXorContext ------------------------------------------------------------------

std::vector<PlanParser::ExprContext *> PlanParser::BitXorContext::expr() {
  return getRuleContexts<PlanParser::ExprContext>();
}

PlanParser::ExprContext* PlanParser::BitXorContext::expr(size_t i) {
  return getRuleContext<PlanParser::ExprContext>(i);
}

tree::TerminalNode* PlanParser::BitXorContext::BXOR() {
  return getToken(PlanParser::BXOR, 0);
}

PlanParser::BitXorContext::BitXorContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::BitXorContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitBitXor(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ExistsContext ------------------------------------------------------------------

tree::TerminalNode* PlanParser::ExistsContext::EXISTS() {
  return getToken(PlanParser::EXISTS, 0);
}

PlanParser::ExprContext* PlanParser::ExistsContext::expr() {
  return getRuleContext<PlanParser::ExprContext>(0);
}

PlanParser::ExistsContext::ExistsContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::ExistsContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitExists(this);
  else
    return visitor->visitChildren(this);
}
//----------------- BitAndContext ------------------------------------------------------------------

std::vector<PlanParser::ExprContext *> PlanParser::BitAndContext::expr() {
  return getRuleContexts<PlanParser::ExprContext>();
}

PlanParser::ExprContext* PlanParser::BitAndContext::expr(size_t i) {
  return getRuleContext<PlanParser::ExprContext>(i);
}

tree::TerminalNode* PlanParser::BitAndContext::BAND() {
  return getToken(PlanParser::BAND, 0);
}

PlanParser::BitAndContext::BitAndContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::BitAndContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitBitAnd(this);
  else
    return visitor->visitChildren(this);
}
//----------------- EmptyTermContext ------------------------------------------------------------------

PlanParser::ExprContext* PlanParser::EmptyTermContext::expr() {
  return getRuleContext<PlanParser::ExprContext>(0);
}

tree::TerminalNode* PlanParser::EmptyTermContext::EmptyTerm() {
  return getToken(PlanParser::EmptyTerm, 0);
}

tree::TerminalNode* PlanParser::EmptyTermContext::IN() {
  return getToken(PlanParser::IN, 0);
}

tree::TerminalNode* PlanParser::EmptyTermContext::NIN() {
  return getToken(PlanParser::NIN, 0);
}

PlanParser::EmptyTermContext::EmptyTermContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::EmptyTermContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitEmptyTerm(this);
  else
    return visitor->visitChildren(this);
}
//----------------- PowerContext ------------------------------------------------------------------

std::vector<PlanParser::ExprContext *> PlanParser::PowerContext::expr() {
  return getRuleContexts<PlanParser::ExprContext>();
}

PlanParser::ExprContext* PlanParser::PowerContext::expr(size_t i) {
  return getRuleContext<PlanParser::ExprContext>(i);
}

tree::TerminalNode* PlanParser::PowerContext::POW() {
  return getToken(PlanParser::POW, 0);
}

PlanParser::PowerContext::PowerContext(ExprContext *ctx) { copyFrom(ctx); }


std::any PlanParser::PowerContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<PlanVisitor*>(visitor))
    return parserVisitor->visitPower(this);
  else
    return visitor->visitChildren(this);
}

PlanParser::ExprContext* PlanParser::expr() {
   return expr(0);
}

PlanParser::ExprContext* PlanParser::expr(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  PlanParser::ExprContext *_localctx = _tracker.createInstance<ExprContext>(_ctx, parentState);
  PlanParser::ExprContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 0;
  enterRecursionRule(_localctx, 0, PlanParser::RuleExpr, precedence);

    size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(56);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case PlanParser::IntegerConstant: {
        _localctx = _tracker.createInstance<IntegerContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;

        setState(3);
        match(PlanParser::IntegerConstant);
        break;
      }

      case PlanParser::FloatingConstant: {
        _localctx = _tracker.createInstance<FloatingContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(4);
        match(PlanParser::FloatingConstant);
        break;
      }

      case PlanParser::BooleanConstant: {
        _localctx = _tracker.createInstance<BooleanContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(5);
        match(PlanParser::BooleanConstant);
        break;
      }

      case PlanParser::StringLiteral: {
        _localctx = _tracker.createInstance<StringContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(6);
        match(PlanParser::StringLiteral);
        break;
      }

      case PlanParser::Identifier: {
        _localctx = _tracker.createInstance<IdentifierContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(7);
        match(PlanParser::Identifier);
        break;
      }

      case PlanParser::JSONIdentifier: {
        _localctx = _tracker.createInstance<JSONIdentifierContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(8);
        match(PlanParser::JSONIdentifier);
        break;
      }

      case PlanParser::T__0: {
        _localctx = _tracker.createInstance<ParensContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(9);
        match(PlanParser::T__0);
        setState(10);
        expr(0);
        setState(11);
        match(PlanParser::T__1);
        break;
      }

      case PlanParser::T__2: {
        _localctx = _tracker.createInstance<ArrayContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(13);
        match(PlanParser::T__2);
        setState(14);
        expr(0);
        setState(19);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 0, _ctx);
        while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
          if (alt == 1) {
            setState(15);
            match(PlanParser::T__3);
            setState(16);
            expr(0); 
          }
          setState(21);
          _errHandler->sync(this);
          alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 0, _ctx);
        }
        setState(23);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == PlanParser::T__3) {
          setState(22);
          match(PlanParser::T__3);
        }
        setState(25);
        match(PlanParser::T__4);
        break;
      }

      case PlanParser::ADD:
      case PlanParser::SUB:
      case PlanParser::BNOT:
      case PlanParser::NOT: {
        _localctx = _tracker.createInstance<UnaryContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(27);
        antlrcpp::downCast<UnaryContext *>(_localctx)->op = _input->LT(1);
        _la = _input->LA(1);
        if (!((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & 402702336) != 0))) {
          antlrcpp::downCast<UnaryContext *>(_localctx)->op = _errHandler->recoverInline(this);
        }
        else {
          _errHandler->reportMatch(this);
          consume();
        }
        setState(28);
        expr(20);
        break;
      }

      case PlanParser::JSONContains:
      case PlanParser::ArrayContains: {
        _localctx = _tracker.createInstance<JSONContainsContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(29);
        _la = _input->LA(1);
        if (!(_la == PlanParser::JSONContains

        || _la == PlanParser::ArrayContains)) {
        _errHandler->recoverInline(this);
        }
        else {
          _errHandler->reportMatch(this);
          consume();
        }
        setState(30);
        match(PlanParser::T__0);
        setState(31);
        expr(0);
        setState(32);
        match(PlanParser::T__3);
        setState(33);
        expr(0);
        setState(34);
        match(PlanParser::T__1);
        break;
      }

      case PlanParser::JSONContainsAll:
      case PlanParser::ArrayContainsAll: {
        _localctx = _tracker.createInstance<JSONContainsAllContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(36);
        _la = _input->LA(1);
        if (!(_la == PlanParser::JSONContainsAll

        || _la == PlanParser::ArrayContainsAll)) {
        _errHandler->recoverInline(this);
        }
        else {
          _errHandler->reportMatch(this);
          consume();
        }
        setState(37);
        match(PlanParser::T__0);
        setState(38);
        expr(0);
        setState(39);
        match(PlanParser::T__3);
        setState(40);
        expr(0);
        setState(41);
        match(PlanParser::T__1);
        break;
      }

      case PlanParser::JSONContainsAny:
      case PlanParser::ArrayContainsAny: {
        _localctx = _tracker.createInstance<JSONContainsAnyContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(43);
        _la = _input->LA(1);
        if (!(_la == PlanParser::JSONContainsAny

        || _la == PlanParser::ArrayContainsAny)) {
        _errHandler->recoverInline(this);
        }
        else {
          _errHandler->reportMatch(this);
          consume();
        }
        setState(44);
        match(PlanParser::T__0);
        setState(45);
        expr(0);
        setState(46);
        match(PlanParser::T__3);
        setState(47);
        expr(0);
        setState(48);
        match(PlanParser::T__1);
        break;
      }

      case PlanParser::ArrayLength: {
        _localctx = _tracker.createInstance<ArrayLengthContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(50);
        match(PlanParser::ArrayLength);
        setState(51);
        match(PlanParser::T__0);
        setState(52);
        _la = _input->LA(1);
        if (!(_la == PlanParser::Identifier

        || _la == PlanParser::JSONIdentifier)) {
        _errHandler->recoverInline(this);
        }
        else {
          _errHandler->reportMatch(this);
          consume();
        }
        setState(53);
        match(PlanParser::T__1);
        break;
      }

      case PlanParser::EXISTS: {
        _localctx = _tracker.createInstance<ExistsContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(54);
        match(PlanParser::EXISTS);
        setState(55);
        expr(1);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    _ctx->stop = _input->LT(-1);
    setState(125);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 6, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(123);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 5, _ctx)) {
        case 1: {
          auto newContext = _tracker.createInstance<PowerContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpr);
          setState(58);

          if (!(precpred(_ctx, 21))) throw FailedPredicateException(this, "precpred(_ctx, 21)");
          setState(59);
          match(PlanParser::POW);
          setState(60);
          expr(22);
          break;
        }

        case 2: {
          auto newContext = _tracker.createInstance<MulDivModContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpr);
          setState(61);

          if (!(precpred(_ctx, 19))) throw FailedPredicateException(this, "precpred(_ctx, 19)");
          setState(62);
          antlrcpp::downCast<MulDivModContext *>(_localctx)->op = _input->LT(1);
          _la = _input->LA(1);
          if (!((((_la & ~ 0x3fULL) == 0) &&
            ((1ULL << _la) & 458752) != 0))) {
            antlrcpp::downCast<MulDivModContext *>(_localctx)->op = _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(63);
          expr(20);
          break;
        }

        case 3: {
          auto newContext = _tracker.createInstance<AddSubContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpr);
          setState(64);

          if (!(precpred(_ctx, 18))) throw FailedPredicateException(this, "precpred(_ctx, 18)");
          setState(65);
          antlrcpp::downCast<AddSubContext *>(_localctx)->op = _input->LT(1);
          _la = _input->LA(1);
          if (!(_la == PlanParser::ADD

          || _la == PlanParser::SUB)) {
            antlrcpp::downCast<AddSubContext *>(_localctx)->op = _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(66);
          expr(19);
          break;
        }

        case 4: {
          auto newContext = _tracker.createInstance<ShiftContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpr);
          setState(67);

          if (!(precpred(_ctx, 17))) throw FailedPredicateException(this, "precpred(_ctx, 17)");
          setState(68);
          antlrcpp::downCast<ShiftContext *>(_localctx)->op = _input->LT(1);
          _la = _input->LA(1);
          if (!(_la == PlanParser::SHL

          || _la == PlanParser::SHR)) {
            antlrcpp::downCast<ShiftContext *>(_localctx)->op = _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(69);
          expr(18);
          break;
        }

        case 5: {
          auto newContext = _tracker.createInstance<RangeContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpr);
          setState(70);

          if (!(precpred(_ctx, 10))) throw FailedPredicateException(this, "precpred(_ctx, 10)");
          setState(71);
          antlrcpp::downCast<RangeContext *>(_localctx)->op1 = _input->LT(1);
          _la = _input->LA(1);
          if (!(_la == PlanParser::LT

          || _la == PlanParser::LE)) {
            antlrcpp::downCast<RangeContext *>(_localctx)->op1 = _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(72);
          _la = _input->LA(1);
          if (!(_la == PlanParser::Identifier

          || _la == PlanParser::JSONIdentifier)) {
          _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(73);
          antlrcpp::downCast<RangeContext *>(_localctx)->op2 = _input->LT(1);
          _la = _input->LA(1);
          if (!(_la == PlanParser::LT

          || _la == PlanParser::LE)) {
            antlrcpp::downCast<RangeContext *>(_localctx)->op2 = _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(74);
          expr(11);
          break;
        }

        case 6: {
          auto newContext = _tracker.createInstance<ReverseRangeContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpr);
          setState(75);

          if (!(precpred(_ctx, 9))) throw FailedPredicateException(this, "precpred(_ctx, 9)");
          setState(76);
          antlrcpp::downCast<ReverseRangeContext *>(_localctx)->op1 = _input->LT(1);
          _la = _input->LA(1);
          if (!(_la == PlanParser::GT

          || _la == PlanParser::GE)) {
            antlrcpp::downCast<ReverseRangeContext *>(_localctx)->op1 = _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(77);
          _la = _input->LA(1);
          if (!(_la == PlanParser::Identifier

          || _la == PlanParser::JSONIdentifier)) {
          _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(78);
          antlrcpp::downCast<ReverseRangeContext *>(_localctx)->op2 = _input->LT(1);
          _la = _input->LA(1);
          if (!(_la == PlanParser::GT

          || _la == PlanParser::GE)) {
            antlrcpp::downCast<ReverseRangeContext *>(_localctx)->op2 = _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(79);
          expr(10);
          break;
        }

        case 7: {
          auto newContext = _tracker.createInstance<RelationalContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpr);
          setState(80);

          if (!(precpred(_ctx, 8))) throw FailedPredicateException(this, "precpred(_ctx, 8)");
          setState(81);
          antlrcpp::downCast<RelationalContext *>(_localctx)->op = _input->LT(1);
          _la = _input->LA(1);
          if (!((((_la & ~ 0x3fULL) == 0) &&
            ((1ULL << _la) & 960) != 0))) {
            antlrcpp::downCast<RelationalContext *>(_localctx)->op = _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(82);
          expr(9);
          break;
        }

        case 8: {
          auto newContext = _tracker.createInstance<EqualityContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpr);
          setState(83);

          if (!(precpred(_ctx, 7))) throw FailedPredicateException(this, "precpred(_ctx, 7)");
          setState(84);
          antlrcpp::downCast<EqualityContext *>(_localctx)->op = _input->LT(1);
          _la = _input->LA(1);
          if (!(_la == PlanParser::EQ

          || _la == PlanParser::NE)) {
            antlrcpp::downCast<EqualityContext *>(_localctx)->op = _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(85);
          expr(8);
          break;
        }

        case 9: {
          auto newContext = _tracker.createInstance<BitAndContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpr);
          setState(86);

          if (!(precpred(_ctx, 6))) throw FailedPredicateException(this, "precpred(_ctx, 6)");
          setState(87);
          match(PlanParser::BAND);
          setState(88);
          expr(7);
          break;
        }

        case 10: {
          auto newContext = _tracker.createInstance<BitXorContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpr);
          setState(89);

          if (!(precpred(_ctx, 5))) throw FailedPredicateException(this, "precpred(_ctx, 5)");
          setState(90);
          match(PlanParser::BXOR);
          setState(91);
          expr(6);
          break;
        }

        case 11: {
          auto newContext = _tracker.createInstance<BitOrContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpr);
          setState(92);

          if (!(precpred(_ctx, 4))) throw FailedPredicateException(this, "precpred(_ctx, 4)");
          setState(93);
          match(PlanParser::BOR);
          setState(94);
          expr(5);
          break;
        }

        case 12: {
          auto newContext = _tracker.createInstance<LogicalAndContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpr);
          setState(95);

          if (!(precpred(_ctx, 3))) throw FailedPredicateException(this, "precpred(_ctx, 3)");
          setState(96);
          match(PlanParser::AND);
          setState(97);
          expr(4);
          break;
        }

        case 13: {
          auto newContext = _tracker.createInstance<LogicalOrContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpr);
          setState(98);

          if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
          setState(99);
          match(PlanParser::OR);
          setState(100);
          expr(3);
          break;
        }

        case 14: {
          auto newContext = _tracker.createInstance<LikeContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpr);
          setState(101);

          if (!(precpred(_ctx, 22))) throw FailedPredicateException(this, "precpred(_ctx, 22)");
          setState(102);
          match(PlanParser::LIKE);
          setState(103);
          match(PlanParser::StringLiteral);
          break;
        }

        case 15: {
          auto newContext = _tracker.createInstance<TermContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpr);
          setState(104);

          if (!(precpred(_ctx, 16))) throw FailedPredicateException(this, "precpred(_ctx, 16)");
          setState(105);
          antlrcpp::downCast<TermContext *>(_localctx)->op = _input->LT(1);
          _la = _input->LA(1);
          if (!(_la == PlanParser::IN

          || _la == PlanParser::NIN)) {
            antlrcpp::downCast<TermContext *>(_localctx)->op = _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }

          setState(106);
          match(PlanParser::T__2);
          setState(107);
          expr(0);
          setState(112);
          _errHandler->sync(this);
          alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 3, _ctx);
          while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
            if (alt == 1) {
              setState(108);
              match(PlanParser::T__3);
              setState(109);
              expr(0); 
            }
            setState(114);
            _errHandler->sync(this);
            alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 3, _ctx);
          }
          setState(116);
          _errHandler->sync(this);

          _la = _input->LA(1);
          if (_la == PlanParser::T__3) {
            setState(115);
            match(PlanParser::T__3);
          }
          setState(118);
          match(PlanParser::T__4);
          break;
        }

        case 16: {
          auto newContext = _tracker.createInstance<EmptyTermContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleExpr);
          setState(120);

          if (!(precpred(_ctx, 15))) throw FailedPredicateException(this, "precpred(_ctx, 15)");
          setState(121);
          antlrcpp::downCast<EmptyTermContext *>(_localctx)->op = _input->LT(1);
          _la = _input->LA(1);
          if (!(_la == PlanParser::IN

          || _la == PlanParser::NIN)) {
            antlrcpp::downCast<EmptyTermContext *>(_localctx)->op = _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(122);
          match(PlanParser::EmptyTerm);
          break;
        }

        default:
          break;
        } 
      }
      setState(127);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 6, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

bool PlanParser::sempred(RuleContext *context, size_t ruleIndex, size_t predicateIndex) {
  switch (ruleIndex) {
    case 0: return exprSempred(antlrcpp::downCast<ExprContext *>(context), predicateIndex);

  default:
    break;
  }
  return true;
}

bool PlanParser::exprSempred(ExprContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 0: return precpred(_ctx, 21);
    case 1: return precpred(_ctx, 19);
    case 2: return precpred(_ctx, 18);
    case 3: return precpred(_ctx, 17);
    case 4: return precpred(_ctx, 10);
    case 5: return precpred(_ctx, 9);
    case 6: return precpred(_ctx, 8);
    case 7: return precpred(_ctx, 7);
    case 8: return precpred(_ctx, 6);
    case 9: return precpred(_ctx, 5);
    case 10: return precpred(_ctx, 4);
    case 11: return precpred(_ctx, 3);
    case 12: return precpred(_ctx, 2);
    case 13: return precpred(_ctx, 22);
    case 14: return precpred(_ctx, 16);
    case 15: return precpred(_ctx, 15);

  default:
    break;
  }
  return true;
}

void PlanParser::initialize() {
#if ANTLR4_USE_THREAD_LOCAL_CACHE
  planParserInitialize();
#else
  ::antlr4::internal::call_once(planParserOnceFlag, planParserInitialize);
#endif
}
