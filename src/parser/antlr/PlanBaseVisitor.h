
// Generated from Plan.g4 by ANTLR 4.13.1

#pragma once


#include "antlr4-runtime.h"
#include "PlanVisitor.h"


/**
 * This class provides an empty implementation of PlanVisitor, which can be
 * extended to create a visitor which only needs to handle a subset of the available methods.
 */
class  PlanBaseVisitor : public PlanVisitor {
public:

  virtual std::any visitJSONIdentifier(PlanParser::JSONIdentifierContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitParens(PlanParser::ParensContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitString(PlanParser::StringContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFloating(PlanParser::FloatingContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitJSONContainsAll(PlanParser::JSONContainsAllContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLogicalOr(PlanParser::LogicalOrContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMulDivMod(PlanParser::MulDivModContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitIdentifier(PlanParser::IdentifierContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLike(PlanParser::LikeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLogicalAnd(PlanParser::LogicalAndContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitEquality(PlanParser::EqualityContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBoolean(PlanParser::BooleanContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitShift(PlanParser::ShiftContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitReverseRange(PlanParser::ReverseRangeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBitOr(PlanParser::BitOrContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAddSub(PlanParser::AddSubContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRelational(PlanParser::RelationalContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitArrayLength(PlanParser::ArrayLengthContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTerm(PlanParser::TermContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitJSONContains(PlanParser::JSONContainsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRange(PlanParser::RangeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitUnary(PlanParser::UnaryContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitInteger(PlanParser::IntegerContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitArray(PlanParser::ArrayContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitJSONContainsAny(PlanParser::JSONContainsAnyContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBitXor(PlanParser::BitXorContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitExists(PlanParser::ExistsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBitAnd(PlanParser::BitAndContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitEmptyTerm(PlanParser::EmptyTermContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPower(PlanParser::PowerContext *ctx) override {
    return visitChildren(ctx);
  }


};

