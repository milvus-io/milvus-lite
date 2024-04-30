#include "parser.h"
namespace milvus::local {

std::string
ParserToMessage(milvus::proto::schema::CollectionSchema& schema,
                const std::string& exprstr) {
    antlr4::ANTLRInputStream input(exprstr);
    PlanLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    PlanParser parser(&tokens);

    PlanParser::ExprContext* tree = parser.expr();

    auto helper = milvus::local::CreateSchemaHelper(&schema);

    milvus::local::PlanCCVisitor visitor(&helper);
    auto res = std::any_cast<milvus::local::ExprWithDtype>(visitor.visit(tree));
    return res.expr->SerializeAsString();
}

std::shared_ptr<milvus::proto::plan::Expr>
ParseIdentifier(milvus::local::SchemaHelper helper,
                const std::string& identifier) {
    auto expr =
        google::protobuf::Arena::CreateMessage<milvus::proto::plan::Expr>(NULL);

    assert(expr->column_expr().has_info());
    expr->ParseFromString(ParserToMessage(*(helper.schema), identifier));

    auto ret = std::make_shared<milvus::proto::plan::Expr>();
    ret.reset(expr);
    return ret;
};

}  // namespace milvus::local
